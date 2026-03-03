"""
map_to_tickers.py
-----------------
Match H1B LCA employer names to their US-listed stock ticker.

Output: ticker_mapping.csv  (one row per employer with >= MIN_PETITIONS petitions)
  EMPLOYER_NORM    – uppercased employer name as it appears in DOL LCA data
  TICKER           – stock ticker (empty = not found / not publicly traded in the US)
  EXCHANGE         – NASDAQ / NYSE / NYSE ARCA / etc.
  LISTED_NAME      – official company name from Yahoo Finance
  MATCH_TYPE       – manual | listing | search | not_traded | not_found
  MATCH_SCORE      – 0-100 fuzzy similarity score (100 = exact manual rule)
  TOTAL_PETITIONS  – petition count summed across all loaded LCA files

Matching strategy (applied in order; first match wins):
  1. Manual patterns  – curated substring rules for subsidiaries, rebrands, known-private
  2. NASDAQ listing   – fuzzy match against nasdaqlisted.txt
  3. yfinance search  – Yahoo Finance for employers >= YF_MIN_PETITIONS not yet matched

Refresh / update workflow:
  Run the same command after adding new LCA files or updating MANUAL_PATTERNS.
  Previously confirmed matches (manual / listing / search / not_traded) are reused;
  only new employers and still-unresolved (not_found) rows are re-processed.

Usage:
  python map_to_tickers.py
"""

import re
import sys
import time
import pandas as pd
from pathlib import Path

from rapidfuzz import fuzz, process as rfprocess
import yfinance as yf

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CSV_DIR          = Path("csv")           # directory containing lca_disclosure_fy*.csv
OUT_FILE         = Path("ticker_mapping.csv")
NASDAQ_FILE      = Path("nasdaqlisted.txt")
MIN_PETITIONS    = 10   # employers below this threshold are excluded from the output
FUZZY_MIN        = 84   # fuzzy score cutoff for employers >= YF_MIN_PETITIONS
FUZZY_MIN_SMALL  = 92   # stricter cutoff for small employers (fewer false positives)
YF_MIN_PETITIONS = 50   # only run yfinance search for employers with >= this many petitions

# US exchange codes returned by Yahoo Finance / yfinance
US_EXCHANGES = {
    'NMS', 'NGS', 'NCM',        # NASDAQ Global Select / Global / Capital
    'NYQ', 'NYE',               # NYSE
    'PCX',                      # NYSE ARCA
    'ASE',                      # NYSE AMEX
    'BTS',                      # CBOE
}

# ---------------------------------------------------------------------------
# Manual patterns
# Checked in order; first match wins (substring match on uppercased H1B name).
# ticker = ""  → confirmed not publicly traded in the US
# ---------------------------------------------------------------------------

MANUAL_PATTERNS = [
    # --- Amazon entities → AMZN ---
    ("AMAZON.COM SERVICES",             "AMZN"),
    ("AMAZON WEB SERVICES",             "AMZN"),
    ("AMAZON DEVELOPMENT CENTER",       "AMZN"),
    ("AMAZON DATA SERVICES",            "AMZN"),

    # --- Alphabet / Google entities → GOOGL ---
    ("GOOGLE LLC",                      "GOOGL"),
    ("GOOGLE INC",                      "GOOGL"),
    ("ALPHABET INC",                    "GOOGL"),
    ("YOUTUBE",                         "GOOGL"),
    ("DEEPMIND",                        "GOOGL"),

    # --- Meta / Facebook rebranding → META ---
    ("META PLATFORMS",                  "META"),
    ("FACEBOOK, INC",                   "META"),
    ("FACEBOOK INC",                    "META"),
    ("INSTAGRAM",                       "META"),
    ("WHATSAPP",                        "META"),

    # --- Microsoft subsidiaries → MSFT ---
    ("LINKEDIN CORPORATION",            "MSFT"),
    ("GITHUB",                          "MSFT"),

    # --- Oracle entities → ORCL ---
    ("ORACLE AMERICA",                  "ORCL"),

    # --- Salesforce entities → CRM ---
    ("SALESFORCE.COM",                  "CRM"),
    ("SALESFORCE, INC",                 "CRM"),

    # --- Goldman Sachs entities → GS ---
    ("GOLDMAN SACHS & CO",              "GS"),
    ("GOLDMAN SACHS SERVICES",          "GS"),

    # --- Walmart entities → WMT ---
    ("WAL-MART",                        "WMT"),
    ("WALMART",                         "WMT"),

    # --- JPMorgan entities → JPM ---
    ("JPMORGAN CHASE",                  "JPM"),

    # --- Qualcomm entities → QCOM ---
    ("QUALCOMM TECHNOLOGIES",           "QCOM"),

    # --- Visa entity → V ---
    ("VISA TECHNOLOGY & OPERATIONS",    "V"),
    ("VISA U.S.A",                      "V"),

    # --- Capital One entity → COF ---
    ("CAPITAL ONE SERVICES",            "COF"),

    # --- Bank of America entity → BAC ---
    ("BANK OF AMERICA",                 "BAC"),

    # --- Wells Fargo entity → WFC ---
    ("WELLS FARGO BANK",                "WFC"),

    # --- Citibank / Citigroup → C ---
    ("CITIBANK",                        "C"),
    ("CITIGROUP",                       "C"),

    # --- US Bancorp entity → USB ---
    ("U.S. BANK NATIONAL ASSOCIATION",  "USB"),
    ("US BANK NATIONAL ASSOCIATION",    "USB"),

    # --- American Express → AXP ---
    ("AMERICAN EXPRESS",                "AXP"),

    # --- Optum → UnitedHealth (UNH) ---
    ("OPTUM SERVICES",                  "UNH"),

    # --- HP split → HPE / HPQ ---
    ("HEWLETT PACKARD ENTERPRISE",      "HPE"),
    ("HP INC",                          "HPQ"),

    # --- VMware acquired by Broadcom → AVGO ---
    ("VMWARE",                          "AVGO"),

    # --- Discover Financial → DFS ---
    ("DFS CORPORATE SERVICES",          "DFS"),

    # --- Cognizant → CTSH ---
    ("COGNIZANT",                       "CTSH"),

    # --- Accenture → ACN ---
    ("ACCENTURE",                       "ACN"),

    # --- IBM entities → IBM ---
    ("IBM CORPORATION",                 "IBM"),
    ("INTERNATIONAL BUSINESS MACHINES", "IBM"),

    # --- T-Mobile → TMUS ---
    ("T-MOBILE",                        "TMUS"),

    # --- Comcast → CMCSA ---
    ("COMCAST",                         "CMCSA"),

    # --- BlackRock → BLK ---
    ("BLACKROCK",                       "BLK"),

    # --- Mastercard → MA ---
    ("MASTERCARD",                      "MA"),

    # --- AT&T → T ---
    ("AT&T",                            "T"),

    # --- Western Digital → WDC ---
    ("WESTERN DIGITAL",                 "WDC"),

    # --- Barclays → BCS (NYSE ADR) ---
    ("BARCLAYS",                        "BCS"),

    # --- FIS → FIS (NYSE) ---
    ("FIS MANAGEMENT SERVICES",         "FIS"),

    # --- CGI → GIB (NYSE) ---
    ("CGI TECHNOLOGIES",                "GIB"),

    # --- SAP (NYSE ADR for SAP SE) → SAP ---
    ("SAP AMERICA",                     "SAP"),

    # --- Anthem / Elevance Health (renamed Jun 2022) → ELV ---
    ("ANTHEM",                          "ELV"),

    # --- CVS Health entities → CVS ---
    ("AETNA",                           "CVS"),   # Aetna acquired by CVS Health 2018
    ("CAREMARK",                        "CVS"),
    ("CVS PHARMACY",                    "CVS"),

    # --- ADP → ADP ---
    ("ADP TECHNOLOGY SERVICES",         "ADP"),

    # --- State Street → STT ---
    ("STATE STREET",                    "STT"),

    # --- Verizon entities → VZ ---
    ("VERIZON DATA SERVICES",           "VZ"),

    # --- Truist Financial → TFC ---
    ("TRUIST",                          "TFC"),

    # --- Nokia (NYSE ADR) → NOK ---
    ("NOKIA OF AMERICA",                "NOK"),

    # --- Charles Schwab → SCHW ---
    ("CHARLES SCHWAB",                  "SCHW"),

    # --- Goldman Sachs bank subsidiary → GS ---
    ("GOLDMAN SACHS BANK",              "GS"),

    # --- ASML (NASDAQ ADR) → ASML ---
    ("ASML US",                         "ASML"),

    # --- FedEx entities → FDX ---
    ("FEDEX CORPORATE SERVICES",        "FDX"),
    ("FEDERAL EXPRESS",                 "FDX"),

    # --- Dell Technologies entities → DELL ---
    ("DELL USA",                        "DELL"),
    ("DELL PRODUCTS",                   "DELL"),
    ("EMC CORPORATION",                 "DELL"),   # EMC merged into Dell 2016

    # --- Qualcomm subsidiary → QCOM ---
    ("QUALCOMM ATHEROS",                "QCOM"),

    # --- Intuitive Surgical → ISRG ---
    ("INTUITIVE SURGICAL",              "ISRG"),

    # --- Amgen → AMGN ---
    ("AMGEN",                           "AMGN"),

    # --- Deutsche Bank (NYSE ADR) → DB ---
    ("DB USA CORE",                     "DB"),
    ("DB GLOBAL TECHNOLOGY",            "DB"),

    # --- Discover Financial entities → DFS (acquired by Capital One May 2024) ---
    ("DISCOVER PRODUCTS",               "DFS"),

    # --- Capital One national bank entity → COF ---
    ("CAPITAL ONE, NATIONAL",           "COF"),

    # --- Marvell Technology → MRVL ---
    ("MARVELL SEMICONDUCTOR",           "MRVL"),

    # --- NXP Semiconductors (NASDAQ) → NXPI ---
    ("NXP USA",                         "NXPI"),

    # --- Starbucks → SBUX ---
    ("STARBUCKS",                       "SBUX"),

    # --- Northern Trust → NTRS ---
    ("NORTHERN TRUST",                  "NTRS"),

    # --- Intercontinental Exchange → ICE ---
    ("INTERCONTINENTAL EXCHANGE",       "ICE"),

    # --- EXL Service (NASDAQ) → EXLS ---
    ("EXLSERVICE",                      "EXLS"),

    # --- Lucid Group (NASDAQ) → LCID ---
    ("LUCID USA",                       "LCID"),

    # --- Target (subsidiary entity) → TGT ---
    ("TARGET ENTERPRISE",               "TGT"),

    # --- UBS (NYSE ADR) → UBS ---
    ("UBS BUSINESS SOLUTIONS",          "UBS"),
    ("CREDIT SUISSE",                   "UBS"),   # Credit Suisse acquired by UBS 2023

    # --- Sony Group (NYSE ADR) → SONY ---
    ("SONY INTERACTIVE ENTERTAINMENT",  "SONY"),

    # --- Cerner → Oracle (acquired June 2022) → ORCL ---
    ("CERNER CORPORATION",              "ORCL"),

    # --- Xilinx → AMD (acquired February 2022) → AMD ---
    ("XILINX",                          "AMD"),

    # --- IBM India entity → IBM ---
    ("IBM INDIA PRIVATE LIMITED",       "IBM"),

    # --- Waymo → Alphabet (GOOGL) ---
    ("WAYMO",                           "GOOGL"),

    # --- Stellantis (NYSE) → STLA (successor to FCA / Fiat Chrysler) ---
    ("FCA US",                          "STLA"),

    # --- Bank of America securities subsidiary → BAC ---
    ("BOFA SECURITIES",                 "BAC"),

    # --- DXC Technology (CSC + HP Enterprise Services merger 2017) → DXC ---
    ("COMPUTER SCIENCES CORPORATION",   "DXC"),
    ("CSC COVANSYS",                    "DXC"),   # Covansys merged into CSC → DXC

    # --- Juniper Networks → HPE (acquired March 2024) → HPE ---
    ("JUNIPER NETWORKS",                "HPE"),

    # --- Nordstrom → JWN (public for most of data period; went private Mar 2025) ---
    ("NORDSTROM",                       "JWN"),

    # --- GEICO → Berkshire Hathaway → BRK-B ---
    ("GOVERNMENT EMPLOYEES INSURANCE",  "BRK-B"),
    ("GEICO",                           "BRK-B"),

    # --- Safeway → Albertsons (ACI, NYSE) ---
    ("SAFEWAY",                         "ACI"),

    # --- Uber Technologies → UBER ---
    ("UBER TECHNOLOGIES",               "UBER"),

    # --- Citizens Financial Group → CFG (NYSE) ---
    ("CITIZENS FINANCIAL GROUP",        "CFG"),

    # --- PNC Financial Services Group → PNC (NYSE) ---
    ("PNC FINANCIAL SERVICES",          "PNC"),

    # --- AstraZeneca (NASDAQ ADR) → AZN ---
    ("ASTRAZENECA",                     "AZN"),

    # --- Schlumberger → SLB (NYSE) ---
    ("SCHLUMBERGER",                    "SLB"),

    # --- Macy's → M (NYSE) ---
    ("MACY'S SYSTEMS",                  "M"),
    ("MACY'S RETAIL",                   "M"),

    # --- Vertex Pharmaceuticals → VRTX (NASDAQ) ---
    ("VERTEX PHARMACEUTICALS",          "VRTX"),

    # --- PPD → Thermo Fisher Scientific (acquired Dec 2021) → TMO ---
    ("PPD DEVELOPMENT",                 "TMO"),

    # --- Cigna / Evernorth → CI (NYSE) ---
    ("CIGNA",                           "CI"),
    ("EVERNORTH",                       "CI"),

    # --- Sirius XM → SIRI (NASDAQ) ---
    ("SIRIUS XM",                       "SIRI"),

    # --- Paycom → PAYC (NYSE) ---
    ("PAYCOM",                          "PAYC"),

    # --- ANSYS → SNPS (NASDAQ: Synopsys acquired ANSYS in Jan 2025) ---
    ("ANSYS, INC",                      "SNPS"),   # specific to avoid matching COVANSYS

    # --- GoDaddy → GDDY (NYSE) ---
    ("GODADDY",                         "GDDY"),

    # --- Informatica → INFA (NYSE) ---
    ("INFORMATICA",                     "INFA"),

    # --- Moody's → MCO (NYSE) ---
    ("MOODY'S ANALYTICS",               "MCO"),
    ("MOODYS ANALYTICS",                "MCO"),

    # --- AECOM → ACM (NYSE) ---
    ("AECOM",                           "ACM"),

    # --- Aptiv (formerly Delphi Automotive) → APTV (NYSE) ---
    ("APTIV",                           "APTV"),

    # --- Spotify → SPOT (NYSE) ---
    ("SPOTIFY",                         "SPOT"),

    # --- Red Hat → IBM (acquired 2019) ---
    ("RED HAT",                         "IBM"),

    # --- Splunk → Cisco (acquired March 2024) ---
    ("SPLUNK",                          "CSCO"),

    # --- SAP Labs → SAP (subsidiary of SAP SE, NYSE) ---
    ("SAP LABS",                        "SAP"),

    # --- Rockwell Collins → RTX (acquired by United Technologies 2018, became RTX) ---
    ("ROCKWELL COLLINS",                "RTX"),

    # --- Zoox → Amazon (acquired 2020) ---
    ("ZOOX",                            "AMZN"),

    # --- Audible → Amazon (subsidiary) ---
    ("AUDIBLE, INC",                    "AMZN"),

    # --- Deutsche Bank Securities → DB (NYSE ADR) ---
    ("DEUTSCHE BANK SECURITIES",        "DB"),

    # --- Cybersource → Visa (subsidiary) ---
    ("CYBERSOURCE",                     "V"),

    # --- Genentech → not_traded (Roche subsidiary, Swiss-listed; only OTC in US) ---
    ("GENENTECH",                       ""),

    # ===========================================================
    # Additional public-company subsidiaries / entities
    # ===========================================================

    # --- Home Depot entities → HD ---
    ("HOME DEPOT",                      "HD"),

    # --- Jacobs Engineering → J (NYSE) ---
    ("JACOBS ENGINEERING",              "J"),

    # --- Medtronic / Covidien → MDT ---
    ("COVIDIEN",                        "MDT"),

    # --- Royal Bank of Canada (NYSE ADR) → RY ---
    ("RBC CAPITAL MARKETS",             "RY"),

    # --- Centene Corporation → CNC ---
    ("CENTENE",                         "CNC"),

    # --- NCR Voyix (post-2023 split of NCR Corporation) → VYX ---
    ("NCR CORPORATION",                 "VYX"),

    # --- S&P Global → SPGI ---
    ("MARKIT NORTH AMERICA",            "SPGI"),
    ("S&P GLOBAL MARKET INTELLIGENCE",  "SPGI"),
    ("STANDARD & POOR'S",               "SPGI"),

    # --- Danaher subsidiaries → DHR ---
    ("CEPHEID",                         "DHR"),
    ("BECKMAN COULTER",                 "DHR"),

    # --- American International Group → AIG ---
    ("AMERICAN GENERAL LIFE",           "AIG"),
    ("AIG PC GLOBAL",                   "AIG"),
    ("AIG EMPLOYEE SERVICES",           "AIG"),

    # --- John Deere → DE ---
    ("DEERE AND COMPANY",               "DE"),

    # --- EchoStar / DISH → SATS ---
    ("DISH WIRELESS",                   "SATS"),

    # --- IQVIA → IQV ---
    ("IQVIA",                           "IQV"),

    # --- Huntington Bancshares → HBAN ---
    ("HUNTINGTON NATIONAL BANK",        "HBAN"),

    # --- Raymond James Financial → RJF ---
    ("RAYMOND JAMES",                   "RJF"),

    # --- HCA Healthcare → HCA ---
    ("HCA MANAGEMENT",                  "HCA"),

    # --- MUFG (NYSE ADR) → MUFG ---
    ("MUFG UNION BANK",                 "MUFG"),
    ("MUFG BANK",                       "MUFG"),

    # --- Rocket Companies → RKT ---
    ("ROCKET MORTGAGE",                 "RKT"),
    ("QUICKEN LOANS",                   "RKT"),
    ("ROCK CENTRAL",                    "RKT"),

    # --- Verisk Analytics → VRSK ---
    ("INSURANCE SERVICES OFFICE",       "VRSK"),

    # --- Progressive Corp → PGR ---
    ("PROGRESSIVE CASUALTY INSURANCE",  "PGR"),

    # --- TE Connectivity → TEL ---
    ("TE CONNECTIVITY",                 "TEL"),

    # --- The Hartford → HIG ---
    ("HARTFORD FIRE INSURANCE",         "HIG"),

    # --- Ally Financial → ALLY ---
    ("ALLY BANK",                       "ALLY"),

    # --- Merck → MRK ---
    ("MERCK SHARP & DOHME",             "MRK"),

    # --- LendingClub → LC ---
    ("LENDINGCLUB",                     "LC"),

    # --- RELX (LexisNexis parent) → RELX ---
    ("LEXISNEXIS",                      "RELX"),

    # --- T. Rowe Price → TROW ---
    ("T. ROWE PRICE",                   "TROW"),

    # --- Merrill Lynch → BAC (BofA subsidiary) ---
    ("MERRILL LYNCH",                   "BAC"),

    # --- AllianceBernstein → AB ---
    ("ALLIANCEBERNSTEIN",               "AB"),

    # --- Magna International → MGA ---
    ("MAGNA ELECTRONICS",               "MGA"),

    # --- Upstart Holdings → UPST ---
    ("UPSTART NETWORK",                 "UPST"),

    # --- Teradata → TDC ---
    ("TERADATA OPERATIONS",             "TDC"),

    # --- SAP entities → SAP ---
    ("ARIBA",                           "SAP"),
    ("SUCCESSFACTORS",                  "SAP"),
    ("ACUMEN SOLUTIONS",                "SAP"),

    # --- Synchrony Financial → SYF ---
    ("SYNCHRONY BANK",                  "SYF"),

    # --- Bill.com → BILL ---
    ("BILL.COM",                        "BILL"),

    # --- ManpowerGroup (Experis) → MAN ---
    ("EXPERIS US",                      "MAN"),

    # --- CME Group → CME ---
    ("CHICAGO MERCANTILE EXCHANGE",     "CME"),

    # --- Abbott Laboratories → ABT ---
    ("ST. JUDE MEDICAL",                "ABT"),

    # --- Bank of Montreal (NYSE ADR) → BMO ---
    ("BANK OF THE WEST",                "BMO"),

    # --- GSK plc (NYSE ADR) → GSK ---
    ("GLAXOSMITHKLINE",                 "GSK"),

    # --- Santander (NYSE ADR) → SAN ---
    ("SANTANDER BANK",                  "SAN"),
    ("SANTANDER CONSUMER",              "SAN"),

    # --- News Corp → NWS ---
    ("MOVE, INC",                       "NWS"),
    ("DOW JONES AND COMPANY",           "NWS"),
    ("DOW JONES & COMPANY",             "NWS"),

    # --- Halliburton → HAL ---
    ("HALLIBURTON",                     "HAL"),

    # --- Joby Aviation → JOBY ---
    ("JOBY AERO",                       "JOBY"),

    # --- Gen Digital (Norton) → GEN ---
    ("NORTONLIFELOCK",                  "GEN"),
    ("LIFELOCK",                        "GEN"),

    # --- Strategy / MicroStrategy → MSTR ---
    ("MICROSTRATEGY",                   "MSTR"),

    # --- Analog Devices (acquired Maxim 2021) → ADI ---
    ("MAXIM INTEGRATED",                "ADI"),

    # --- Cummins → CMI ---
    ("CUMMINS",                         "CMI"),

    # --- ABB Ltd (NYSE ADR) → ABB ---
    ("ABB INC",                         "ABB"),

    # --- Seagate Technology → STX ---
    ("SEAGATE US",                      "STX"),

    # --- Express Scripts / Evernorth (Cigna) → CI (not duplicate with CIGNA) ---
    ("EXPRESS SCRIPTS",                 "CI"),

    # --- Activision / Blizzard → MSFT (acquired 2023) ---
    ("ACTIVISION PUBLISHING",           "MSFT"),
    ("BLIZZARD ENTERTAINMENT",          "MSFT"),

    # --- General Dynamics → GD ---
    ("GULFSTREAM AEROSPACE",            "GD"),

    # --- Kite Pharma → GILD (acquired by Gilead 2017) ---
    ("KITE PHARMA",                     "GILD"),

    # --- Philips (NYSE ADR) → PHG ---
    ("PHILIPS NORTH AMERICA",           "PHG"),

    # --- Pearson plc (NYSE ADR) → PSO ---
    ("NCS PEARSON",                     "PSO"),

    # --- GE HealthCare → GEHC ---
    ("GE PRECISION HEALTHCARE",         "GEHC"),
    ("GE HEALTHCARE IITS",              "GEHC"),

    # --- Fifth Third Bancorp → FITB ---
    ("FIFTH THIRD BANK",                "FITB"),

    # --- Manulife / John Hancock (NYSE) → MFC ---
    ("JOHN HANCOCK LIFE INSURANCE",     "MFC"),
    ("JOHN HANCOCK",                    "MFC"),

    # --- East West Bancorp → EWBC ---
    ("EAST WEST BANK",                  "EWBC"),

    # --- Travelers Companies → TRV ---
    ("TRAVELERS INDEMNITY",             "TRV"),
    ("TRAVELERS INSURANCE",             "TRV"),

    # --- Assurant → AIZ ---
    ("AMERICAN BANKERS INSURANCE",      "AIZ"),

    # --- Johnson & Johnson → JNJ ---
    ("AURIS HEALTH",                    "JNJ"),

    # --- Toronto-Dominion Bank (NYSE) → TD ---
    ("TD BANK, NATIONAL",               "TD"),
    ("TD SECURITIES",                   "TD"),

    # --- Live Nation → LYV ---
    ("LIVE NATION",                     "LYV"),

    # --- Hilton Hotels → HLT ---
    ("HILTON DOMESTIC OPERATING",       "HLT"),

    # --- TSMC (NYSE ADR) → TSM ---
    ("TSMC ARIZONA",                    "TSM"),

    # --- Lululemon → LULU ---
    ("LULULEMON",                       "LULU"),

    # --- CoStar Group → CSGP ---
    ("COSTAR REALTY",                   "CSGP"),

    # --- Block Inc. (formerly Square; ticker SQ → XYZ May 2024) → XYZ ---
    ("AFTERPAY US",                     "XYZ"),
    ("SQUARE",                          "XYZ"),

    # --- Aegon (Transamerica parent, NYSE ADR) → AEG ---
    ("TRANSAMERICA LIFE INSURANCE",     "AEG"),

    # --- Walt Disney (Hulu parent) → DIS ---
    ("HULU",                            "DIS"),

    # --- BorgWarner → BWA ---
    ("BORGWARNER",                      "BWA"),

    # --- Warner Bros. Discovery → WBD ---
    ("WARNERMEDIA",                     "WBD"),

    # --- National Grid (NYSE ADR) → NGG ---
    ("NATIONAL GRID USA",               "NGG"),

    # --- Comerica → CMA ---
    ("COMERICA MANAGEMENT",             "CMA"),

    # --- BP plc (NYSE) → BP ---
    ("BP AMERICA",                      "BP"),

    # --- Nomura Holdings (NYSE ADR) → NMR ---
    ("NOMURA AMERICA",                  "NMR"),
    ("NOMURA HOLDING AMERICA",          "NMR"),

    # --- Aurora Innovation → AUR ---
    ("AURORA OPERATIONS",               "AUR"),

    # --- Global Business Travel Group → GBTG ---
    ("EGENCIA",                         "GBTG"),
    ("GBT US",                          "GBTG"),

    # --- Allstate → ALL ---
    ("ALLSTATE INSURANCE",              "ALL"),

    # --- Redfin → RKT (acquired by Rocket Companies Aug 2024) ---
    ("REDFIN CORPORATION",              "RKT"),

    # --- Bank of Nova Scotia (NYSE) → BNS ---
    ("SCOTIA CAPITAL",                  "BNS"),

    # --- Corteva Agriscience → CTVA ---
    ("CORTEVA AGRISCIENCE",             "CTVA"),

    # --- Morgan Stanley entities → MS ---
    ("MORGAN STANLEY SMITH BARNEY",     "MS"),

    # --- Regions Financial → RF ---
    ("REGIONS BANK",                    "RF"),

    # --- Parsons Corporation → PSN ---
    ("PARSONS TRANSPORTATION",          "PSN"),

    # --- Fresenius Medical Care (NYSE ADR) → FMS ---
    ("NATIONAL MEDICAL CARE",           "FMS"),
    ("FRESENIUS MEDICAL",               "FMS"),

    # --- Booking Holdings → BKNG ---
    ("PRICELINE",                       "BKNG"),

    # --- Foot Locker → FL ---
    ("FOOT LOCKER CORPORATE",           "FL"),

    # --- WPP plc (NYSE ADR) → WPP ---
    ("GROUPM",                          "WPP"),
    ("GREY GROUP",                      "WPP"),
    ("WUNDERMAN",                       "WPP"),
    ("OGILVY",                          "WPP"),
    ("JWT",                             "WPP"),

    # --- Emerson Electric (acquired NI Corp 2023) → EMR ---
    ("NATIONAL INSTRUMENTS",            "EMR"),

    # --- Voya Financial → VOYA ---
    ("VOYA SERVICES",                   "VOYA"),

    # --- KeyCorp → KEY ---
    ("KEYBANK NATIONAL",                "KEY"),

    # --- NOV Inc. → NOV ---
    ("NATIONAL OILWELL VARCO",          "NOV"),

    # --- Pfizer (acquired Seagen 2023) → PFE ---
    ("SEAGEN",                          "PFE"),

    # --- BeiGene (NASDAQ) → BGNE ---
    ("BEIGENE USA",                     "BGNE"),

    # --- Acuity Brands → AYI ---
    ("ACUITY BRANDS LIGHTING",          "AYI"),

    # --- CNA Financial → CNA ---
    ("CONTINENTAL CASUALTY",            "CNA"),

    # --- Lazard → LAZ ---
    ("LAZARD FRERES",                   "LAZ"),

    # --- AbbVie → ABBV ---
    ("ABBVIE US",                       "ABBV"),

    # --- Adient (automotive seating) → ADNT ---
    ("ADIENT US",                       "ADNT"),

    # --- Paychex (acquired Paycor 2024) → PAYX ---
    ("PAYCOR",                          "PAYX"),

    # --- Stanley Black & Decker → SWK ---
    ("STANLEY BLACK AND DECKER",        "SWK"),
    ("STANLEY BLACK & DECKER",          "SWK"),

    # --- Hyatt Hotels → H ---
    ("HYATT CORPORATION",               "H"),

    # --- NextEra Energy → NEE ---
    ("FLORIDA POWER & LIGHT",           "NEE"),
    ("FPL",                             "NEE"),

    # --- Western Alliance Bancorporation → WAL ---
    ("WESTERN ALLIANCE BANK",           "WAL"),

    # --- OpenText (acquired Micro Focus 2023) → OTEX ---
    ("MICRO FOCUS",                     "OTEX"),

    # --- HSBC Holdings (NYSE ADR) → HSBC ---
    ("HSBC BANK USA",                   "HSBC"),

    # --- LoanDepot → LDI ---
    ("LOANDEPOT",                       "LDI"),

    # --- CIBC (NYSE ADR) → CM ---
    ("CIBC WORLD MARKETS",              "CM"),
    ("CIBC BANK USA",                   "CM"),

    # --- Atos Syntel → Atos (listed in France, not US) ---
    ("ATOS SYNTEL",                     ""),

    # --- Rivian → RIVN ---
    ("RIVIAN AUTOMOTIVE",               "RIVN"),

    # --- ByteDance / TikTok → private ---
    ("BYTEDANCE",                       ""),
    ("TIKTOK",                          ""),

    # --- Mastech → MHH (NASDAQ) ---
    ("MASTECH",                         "MHH"),

    # --- Bloomberg → private ---
    ("BLOOMBERG",                       ""),

    # -----------------------------------------------------------------
    # Known NOT publicly traded in the US
    # -----------------------------------------------------------------
    ("DELOITTE",                        ""),
    ("ERNST & YOUNG",                   ""),
    ("KPMG",                            ""),
    ("PRICEWATERHOUSECOOPERS",          ""),
    ("MCKINSEY & COMPANY",              ""),
    ("BOSTON CONSULTING GROUP",         ""),
    ("BAIN & COMPANY",                  ""),
    ("BAIN AND COMPANY",                ""),
    ("FIDELITY TECHNOLOGY GROUP",       ""),   # Fidelity Investments is private
    ("FIDELITY INVESTMENTS",            ""),
    ("VANGUARD",                        ""),   # mutual structure, not publicly traded
    # Indian IT companies (listed on BSE/NSE, not US exchanges)
    ("TATA CONSULTANCY SERVICES",       ""),
    ("MPHASIS CORPORATION",             ""),
    ("HEXAWARE TECHNOLOGIES",           ""),
    ("TECH MAHINDRA",                   ""),
    ("HCL AMERICA",                     ""),   # HCL Technologies is India-listed
    ("HCL TECHNOLOGIES",                ""),
    ("LTIMINDTREE",                     ""),
    ("LARSEN & TOUBRO INFOTECH",        ""),
    ("L&T TECHNOLOGY SERVICES",         ""),
    ("MINDTREE",                        ""),   # merged into LTIMindtree
    # European / other non-US listed
    ("CAPGEMINI",                       ""),   # Euronext Paris
    # Private staffing / consulting
    ("COMPUNNEL",                       ""),
    ("PEOPLE TECH GROUP",               ""),
    ("GRANDISON MANAGEMENT",            ""),
    ("TEKORG INC",                      ""),
    ("UST GLOBAL",                      ""),
    ("POPULUS GROUP",                   ""),
    ("VIRTUSA CORPORATION",             ""),   # acquired by Cognizant, delisted
    # Non-profit institutions
    ("UNIVERSITY OF",                   ""),
    ("UNIVERSITY AT",                   ""),
    ("STATE UNIVERSITY",                ""),
    ("JOHNS HOPKINS UNIVERSITY",        ""),
    ("STANFORD UNIVERSITY",             ""),
    ("LELAND STANFORD",                 ""),
    ("MAYO CLINIC",                     ""),
    ("CLEVELAND CLINIC",                ""),
    ("KAISER PERMANENTE",               ""),
    # Private / foreign-listed IT & consulting
    ("NTT DATA",                        ""),   # NTT Group, Japan-listed
    ("HCL GLOBAL SYSTEMS",              ""),   # HCL Technologies, India-listed
    ("SYSTEM SOFT TECHNOLOGIES",        ""),
    ("INFINITE COMPUTER SOLUTIONS",     ""),
    ("THE MATHWORKS",                   ""),   # Private (MATLAB / Simulink maker)
    ("V-SOFT CONSULTING",               ""),
    ("CITIUSTECH",                      ""),
    ("SYNECHRON",                       ""),
    ("SKILLTUNE",                       ""),
    ("MANAGEMENT HEALTH SYSTEMS",       ""),
    ("GLOBALLOGIC",                     ""),   # Hitachi subsidiary, Japan-listed
    ("HTC GLOBAL SERVICES",             ""),
    ("ZS ASSOCIATES",                   ""),   # Private management consulting
    ("OATH HOLDINGS",                   ""),   # Verizon Media → Yahoo (private since 2021)
    ("STRIPE",                          ""),   # Private fintech
    ("PROKARMA",                        ""),
    ("DATABRICKS",                      ""),   # Private data/AI platform
    ("INDEED",                          ""),   # Recruit Holdings, Japan-listed
    ("RANDSTAD",                        ""),   # Dutch-listed (Euronext Amsterdam)
    ("BRILLIO",                         ""),
    ("SLALOM",                          ""),   # Private consulting
    ("WSP USA",                         ""),   # WSP Global, Canada (TSX)
    ("SIEMENS INDUSTRY SOFTWARE",       ""),   # Siemens AG, Germany
    ("PERSISTENT SYSTEMS",              ""),   # India-listed (NSE)
    ("BIRLASOFT",                       ""),   # India-listed (NSE)
    ("PERFICIENT",                      ""),   # Private (acquired by EQT 2023)
    ("TWITTER",                         ""),   # Private (X Corp since Oct 2022)
    ("PHOTON INFOTECH",                 ""),
    ("MARLABS",                         ""),
    ("TEK LEADERS",                     ""),
    ("INNOVA SOLUTIONS",                ""),
    ("INTRAEDGE",                       ""),
    ("IDEXCEL",                         ""),
    ("VALUEMOMENTUM",                   ""),
    ("ERP ANALYSTS",                    ""),
    ("NATSOFT",                         ""),
    ("EPITEC",                          ""),
    # Additional non-profits: universities
    ("COLUMBIA UNIVERSITY",             ""),
    ("EMORY UNIVERSITY",                ""),
    ("WASHINGTON UNIVERSITY",           ""),
    ("HARVARD UNIVERSITY",              ""),
    ("NORTHWESTERN UNIVERSITY",         ""),
    ("DUKE UNIVERSITY",                 ""),
    ("NEW YORK UNIVERSITY",             ""),
    ("CORNELL UNIVERSITY",              ""),
    ("PRINCETON UNIVERSITY",            ""),
    ("MASSACHUSETTS INSTITUTE OF TECHNOLOGY", ""),
    # Additional non-profits: hospitals / medical schools
    ("BAYLOR COLLEGE OF MEDICINE",      ""),
    ("WEILL CORNELL",                   ""),
    ("NYU GROSSMAN",                    ""),
    ("BRIGHAM AND WOMEN",               ""),
    ("MEMORIAL SLOAN KETTERING",        ""),
    ("ICAHN SCHOOL OF MEDICINE",        ""),
    ("ST. JUDE CHILDREN",               ""),
    ("NATIONAL INSTITUTES OF HEALTH",   ""),
    ("BATTELLE MEMORIAL",               ""),
    ("UT-BATTELLE",                     ""),
    ("GENERAL HOSPITAL CORPORATION",    ""),   # Mass General Hospital
    ("THE DEVEREUX FOUNDATION",         ""),
    # Government / school districts
    ("DALLAS INDEPENDENT SCHOOL",       ""),
    ("SAVANNAH CHATHAM",                ""),
    # Missing universities
    ("YALE UNIVERSITY",                 ""),
    ("PURDUE UNIVERSITY",               ""),
    ("TEXAS A&M UNIVERSITY",            ""),
    ("INDIANA UNIVERSITY",              ""),
    ("CARNEGIE MELLON UNIVERSITY",      ""),
    ("NORTHEASTERN UNIVERSITY",         ""),
    ("OREGON HEALTH",                   ""),   # Oregon Health & Science University
    ("TEMPLE UNIVERSITY",               ""),
    ("GEORGIA INSTITUTE OF TECHNOLOGY", ""),   # Georgia Tech
    ("CALIFORNIA INSTITUTE OF TECHNOLOGY", ""), # Caltech
    # Non-profit healthcare
    ("UT SOUTHWESTERN",                 ""),
    ("CHILDREN'S HOSPITAL",             ""),
    ("HENRY FORD HEALTH",               ""),
    ("DANA-FARBER CANCER",              ""),
    ("HEALTH CARE SERVICE",             ""),   # BCBS Illinois, private
    ("NORTHWELL HEALTH",                ""),
    ("MONTEFIORE MEDICAL",              ""),
    ("CINCINNATI CHILDREN'S",           ""),
    ("HOWARD HUGHES MEDICAL",           ""),
    # Government labs / research
    ("BROOKHAVEN NATIONAL LABORATORY",  ""),
    ("LAWRENCE BERKELEY NATIONAL",      ""),
    ("MILWAUKEE BOARD OF SCHOOL",       ""),
    # Insurance mutuals / private
    ("NORTHWESTERN MUTUAL",             ""),   # private mutual insurer
    ("PRUDENTIAL INSURANCE",            ""),   # Prudential Financial is PFG, but "PRUDENTIAL INSURANCE COMPANY OF AMERICA" is the NJ mutual; different from PFG
    ("MASSACHUSETTS MUTUAL",            ""),   # MassMutual – private mutual
    ("HOWARD HUGHES MEDICAL",           ""),
    # Private / foreign companies fixing NASDAQ false positives
    ("TAVANT TECHNOLOGIES",             ""),   # Private
    ("KPIT TECHNOLOGIES",               ""),   # India-listed (NSE)
    ("ZENSAR TECHNOLOGIES",             ""),   # India-listed (NSE)
    ("CIGNITI TECHNOLOGIES",            ""),   # India-listed, merged into Coforge
    ("SAMSUNG SEMICONDUCTOR",           ""),   # Korea-listed (KRX)
    ("SAMSUNG AUSTIN SEMICONDUCTOR",    ""),   # Korea-listed
    ("SAMSUNG ELECTRONICS AMERICA",     ""),   # Korea-listed
    ("GP TECHNOLOGIES",                 ""),   # Private staffing
    ("3D TECHNOLOGIES",                 ""),   # Private staffing
    ("CLOUD BIG DATA TECHNOLOGIES",     ""),   # Private staffing
    ("ARTIFINT TECHNOLOGIES",           ""),   # Private staffing
    ("VCLOUD TECHNOLOGY GROUP",         ""),   # Private staffing
    ("YASH TECHNOLOGIES",               ""),   # Private staffing
    ("MACHINE LEARNING TECHNOLOGIES",   ""),   # Private staffing
    ("ROBOTICS TECHNOLOGIES",           ""),   # Private staffing
    ("BLOCKCHAIN TECHNOLOGIES",         ""),   # Private staffing
    ("AUTOMATION TECHNOLOGIES",         ""),   # Private staffing
    ("DATA SCIENCE TECHNOLOGIES",       ""),   # Private staffing
    ("E-GIANTS TECHNOLOGIES",           ""),   # Private staffing
    ("MACHINE INTELLIGENCE TECHNOLOGIES", ""), # Private staffing
    ("DIGITAL TECHNOLOGIES LLC",        ""),   # Private staffing (specific)
    ("TRIAD NATIONAL SECURITY",         ""),   # DOE lab operator
    # Additional private / foreign companies from not_found
    ("HEADSTRONG SERVICES",             ""),   # Capgemini subsidiary
    ("COFORGE",                         ""),   # India-listed (NSE: COFORGE)
    ("ROBERT BOSCH",                    ""),   # Germany-listed
    ("COX AUTOMOTIVE",                  ""),   # Private (Cox Enterprises)
    ("COX ENTERPRISES",                 ""),   # Private
    ("HITACHI VANTARA",                 ""),   # Hitachi subsidiary, Japan-listed
    ("SCHNEIDER ELECTRIC",              ""),   # France-listed (Euronext)
    ("TUSIMPLE",                        ""),   # Delisted 2023
    ("INTELLECTT",                      ""),   # Private staffing
    ("QUEST GLOBAL SERVICES",           ""),   # Private engineering firm
    ("MPG OPERATIONS",                  ""),   # Private
    ("SYNAPSIS INC",                    ""),   # Private (distinct from Synaptics SYNA)
    ("TECHNOSOFT CORPORATION",          ""),   # Private
    ("EFICENS SYSTEMS",                 ""),   # Private
    ("MIRACLE SOFTWARE SYSTEMS",        ""),   # Private
    ("AXTRIA",                          ""),   # Private (pharma analytics)
    ("COLLABORATE SOLUTIONS",           ""),   # Private staffing
    ("XORIANT CORPORATION",             ""),   # Private
    ("QUADRANT RESOURCE",               ""),   # Private staffing
    ("DENKEN SOLUTIONS",                ""),   # Private staffing
    ("CYBERTHINK",                      ""),   # Private staffing
    ("TIGER ANALYTICS",                 ""),   # Private
    ("DIVERSANT",                       ""),   # Private staffing
    ("NAGARRO",                         ""),   # Germany-listed (XETRA: NA9)
    ("ORION SYSTEMS INTEGRATORS",       ""),   # Private
    ("SRS CONSULTING",                  ""),   # Private staffing
    ("INFOGAIN",                        ""),   # Private
    ("MITCHELL/MARTIN",                 ""),   # Private staffing
    ("RELIABLE SOFTWARE RESOURCES",     ""),   # Private staffing
    ("PAMTEN",                          ""),   # Private staffing
    ("INTONE NETWORKS",                 ""),   # Private
    ("SAGE IT",                         ""),   # Private staffing
    ("CONSULTADD",                      ""),   # Private staffing
    ("VASTEK",                          ""),   # Private staffing
    ("VISTA APPLIED SOLUTIONS",         ""),   # Private staffing
    ("HUGHES NETWORK SYSTEMS",          ""),   # EchoStar subsidiary
    ("ECLINICALWORKS",                  ""),   # Private healthcare IT
    ("BLACK & VEATCH",                  ""),   # Private engineering
    ("TOTAL SYSTEM SERVICES",           ""),   # TSYS – now Global Payments subsidiary
    ("CITICORP CREDIT SERVICES",        ""),   # Citi subsidiary (already caught by CITIBANK)
    ("EPSILON DATA MANAGEMENT",         ""),   # Alliance Data / Loyalty One subsidiary
    # Additional not-traded
    ("7-ELEVEN",                        ""),   # Seven & i Holdings (Japan), no major US listing
    ("RSM US",                          ""),   # Private accounting firm
    ("AMERICAN FAMILY MUTUAL",          ""),   # Private mutual insurer
    ("SATIN SOLUTIONS",                 ""),   # Private staffing
    ("ECONTENTI",                       ""),   # Private staffing
    ("DIASPARK",                        ""),   # Private IT services
    ("SLK AMERICA",                     ""),   # Private staffing
    ("PIONEER CONSULTING SERVICES",     ""),   # Private staffing
    ("CYMA SYSTEMS",                    ""),   # Private staffing
    ("OPEN AVENUES FOUNDATION",         ""),   # Non-profit
    ("ASM AMERICA",                     ""),   # ASM International, Dutch-listed (not major US)
    ("ROBOTIC PROCESS AUTOMATION LLC",  ""),   # Generic name, private staffing
    ("3I INFOTECH",                     ""),   # India-listed (BSE/NSE)
    ("CASE WESTERN RESERVE UNIVERSITY", ""),   # Non-profit university
    ("WEST VIRGINIA UNIVERSITY",        ""),   # Public university
    ("TEXAS TECH UNIVERSITY",           ""),   # Public university
    ("GEORGETOWN UNIVERSITY",           ""),   # Non-profit university
    # Remaining not_found – private / foreign / non-profit
    ("VIRTUSA CONSULTING SERVICES",     ""),   # India private-listed entity
    ("SAPIENT CORPORATION",             ""),   # PublicisSapient, Euronext-listed parent
    ("STAPLES, INC",                    ""),   # Went private 2017 (Sycamore Partners)
    ("STAPLES INC",                     ""),
    ("IRIS SOFTWARE",                   ""),   # UK-based, not US-listed
    ("TRUSTEES OF BOSTON UNIVERSITY",   ""),   # Non-profit
    ("DISH NETWORK",                    ""),   # Merged with EchoStar / DirecTV; delisted
    ("ANTRA, INC",                      ""),   # Private staffing
    ("ITC INFOTECH",                    ""),   # ITC Ltd subsidiary, India-listed
    ("VERIDIC SOLUTIONS",               ""),   # Private staffing
    ("INFOCONS INC",                    ""),   # Private staffing
    ("STRATEGIC SYSTEMS",               ""),   # Private staffing / IT consulting
]

# ---------------------------------------------------------------------------
# Name normalisation
# ---------------------------------------------------------------------------

_STRIP_SUFFIXES = re.compile(
    r'\b(LLC|L\.L\.C\.?|LLP|L\.L\.P\.?|INC\.?|CORP\.?|LTD\.?|LIMITED'
    r'|L\.P\.?|PLC|N\.A\.?|N\.V\.?|S\.A\.?|S\.E\.?|A\.G\.?'
    r'|INCORPORATED|CORPORATION|COMPANY|CO\.)\b',
    re.IGNORECASE,
)
_STRIP_LISTING = re.compile(
    # Strips exchange-listing boilerplate from official security names
    r'\s*[-–]\s*(COMMON STOCK|CLASS [A-Z].*|ORDINARY SHARES.*'
    r'|DEPOSITARY.*|ADR|ADS|UNITS|PREFERRED STOCK|WARRANTS?|RIGHTS?).*$',
    re.IGNORECASE,
)
_PUNCT = re.compile(r'[^\w\s]')
_WS    = re.compile(r'\s+')


def normalize(name: str, listing: bool = False) -> str:
    """Normalise a company name for matching."""
    s = str(name).upper()
    if listing:
        s = _STRIP_LISTING.sub('', s)
    s = _STRIP_SUFFIXES.sub(' ', s)
    s = _PUNCT.sub(' ', s)
    s = _WS.sub(' ', s).strip()
    return s

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def compute_petitions() -> dict[str, int]:
    """
    Read all LCA CSV files and return {EMPLOYER_NORM: petition_count}.
    EMPLOYER_NORM = stripped, uppercased EMPLOYER_NAME (no other transformation).
    """
    files = sorted(CSV_DIR.glob('lca_disclosure_fy*.csv'))
    if not files:
        sys.exit(f"[error] no lca_disclosure_fy*.csv found in {CSV_DIR}/")
    frames = [
        pd.read_csv(p, usecols=['EMPLOYER_NAME'], dtype=str, low_memory=False)
        for p in files
    ]
    combined = pd.concat(frames, ignore_index=True)
    combined['EMPLOYER_NORM'] = combined['EMPLOYER_NAME'].fillna('').str.strip().str.upper()
    combined = combined[combined['EMPLOYER_NORM'] != '']
    counts = combined.groupby('EMPLOYER_NORM').size()
    print(f"Loaded {len(files)} LCA file(s): "
          f"{len(counts):,} unique employers, {counts.sum():,} total petitions")
    return counts.to_dict()


def load_existing_lookup() -> dict[str, dict]:
    """
    Load ticker_mapping.csv (if it exists) and return a dict of previously
    confirmed matches keyed by EMPLOYER_NORM.  not_found rows are excluded so
    they can be re-processed with updated patterns.
    """
    if not OUT_FILE.exists():
        return {}
    df = pd.read_csv(OUT_FILE, dtype=str)
    lookup: dict[str, dict] = {}
    for _, row in df.iterrows():
        mt = str(row.get('MATCH_TYPE', '')).strip()
        if mt and mt != 'not_found':
            lookup[str(row['EMPLOYER_NORM']).strip()] = {
                'ticker':      str(row.get('TICKER',      '') or '').strip(),
                'exchange':    str(row.get('EXCHANGE',    '') or '').strip(),
                'listed_name': str(row.get('LISTED_NAME', '') or '').strip(),
                'match_type':  mt,
                'match_score': str(row.get('MATCH_SCORE', '') or '').strip(),
            }
    print(f"Existing mapping: {len(lookup):,} confirmed matches loaded from {OUT_FILE}")
    return lookup

# ---------------------------------------------------------------------------
# NASDAQ listing file
# ---------------------------------------------------------------------------

def load_nasdaq_listings(path: Path) -> tuple[list[str], list[str]]:
    """
    Parse nasdaqlisted.txt (pipe-delimited).  Returns parallel lists:
      (normalised_names, symbols)
    Filters out ETFs, test issues, and the trailing metadata row.
    """
    df = pd.read_csv(path, sep='|', engine='python')
    # Drop trailing metadata row
    df = df[df['Symbol'].notna() & ~df['Symbol'].str.startswith('File')].copy()
    # Keep only real equities (no ETFs, no test issues)
    df = df[(df['ETF'] == 'N') & (df['Test Issue'] == 'N')]
    names   = [normalize(n, listing=True) for n in df['Security Name'].tolist()]
    symbols = df['Symbol'].tolist()
    return names, symbols


def nasdaq_match(employer_norm: str, listing_names: list[str],
                 listing_symbols: list[str],
                 min_score: int = FUZZY_MIN) -> tuple[str, str, int]:
    """Fuzzy match one employer name against the NASDAQ listing."""
    result = rfprocess.extractOne(
        employer_norm, listing_names,
        scorer=fuzz.token_set_ratio,
        score_cutoff=min_score,
    )
    if result:
        _, score, idx = result
        return listing_symbols[idx], 'NASDAQ', int(score)
    return '', '', 0


# ---------------------------------------------------------------------------
# Matching helpers
# ---------------------------------------------------------------------------

def apply_manual(name_upper: str) -> tuple[str | None, str]:
    """Return (ticker, 'manual') or (None, '') – first matching pattern wins."""
    for pattern, ticker in MANUAL_PATTERNS:
        if pattern.upper() in name_upper:
            return ticker, 'manual'
    return None, ''


def _first_us_equity(results: list, name_norm: str) -> tuple[str, str, int]:
    """Pick the best US equity from a yfinance Search result list."""
    for r in results:
        if r.get('quoteType') != 'EQUITY':
            continue
        if r.get('exchange', '') not in US_EXCHANGES:
            continue
        returned_name  = r.get('shortname') or r.get('longname') or ''
        returned_norm  = normalize(returned_name, listing=True)
        score = fuzz.token_set_ratio(name_norm, returned_norm)
        if score < FUZZY_MIN:
            continue
        ticker      = r.get('symbol', '')
        exc_display = r.get('exchDisp') or _EXC_MAP.get(r.get('exchange', ''), r.get('exchange', ''))
        return ticker, exc_display, score
    return '', '', 0


# Pre-compiled pattern to extract the first 3 meaningful words for fallback search
_LEAD_WORDS = re.compile(
    r'^((?:\S+\s+){0,2}\S+)',   # first 1-3 whitespace-separated tokens
)
_SKIP_WORDS = re.compile(
    r'\b(LLC|LLP|INC|CORP|LTD|LIMITED|SERVICES|SOLUTIONS|TECHNOLOGIES|'
    r'TECHNOLOGY|SYSTEMS|CONSULTING|MANAGEMENT|GROUP|GLOBAL|INTERNATIONAL|'
    r'SOFTWARE|DIGITAL|NETWORKS|STAFFING|OUTSOURCING|USA|U\.S\.?|AMERICA)\b',
    re.IGNORECASE,
)


def _short_query(name: str) -> str:
    """Strip noise and take the first 2-3 meaningful tokens as a search query."""
    s = _SKIP_WORDS.sub(' ', name)
    s = re.sub(r'[^\w\s&]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    tokens = s.split()[:3]
    return ' '.join(tokens)


def yf_search(name: str) -> tuple[str, str, int]:
    """
    Search Yahoo Finance for a US-listed equity matching the given employer name.
    Tries the full name first; if no US equity passes the similarity gate, retries
    with a shorter, noise-stripped query.
    Returns (ticker, exchange_display, score) or ('', '', 0) if no match found.
    """
    name_norm = normalize(name)

    for query in [name, _short_query(name)]:
        if not query.strip():
            continue
        try:
            results = yf.Search(query, max_results=8).quotes or []
        except Exception:
            results = []
        ticker, exchange, score = _first_us_equity(results, name_norm)
        if ticker:
            return ticker, exchange, score
        time.sleep(0.1)   # small extra pause between the two attempts

    return '', '', 0

# ---------------------------------------------------------------------------
# yfinance validation
# ---------------------------------------------------------------------------

_EXC_MAP = {
    'NMS': 'NASDAQ', 'NGS': 'NASDAQ', 'NCM': 'NASDAQ',
    'NYQ': 'NYSE',   'NYE': 'NYSE',
    'PCX': 'NYSE ARCA', 'ASE': 'NYSE AMEX',
    'BTS': 'CBOE', 'GTS': 'NYSE ARCA', 'PNK': 'OTC',
}


def validate_tickers(tickers: list[str]) -> dict[str, dict]:
    """
    Check each ticker via yfinance: confirm it has a price, get official name
    and exchange.  Returns {ticker: {'yf_name', 'yf_exchange', 'valid'}}.
    """
    unique = sorted({t for t in tickers if t})
    print(f"  Validating {len(unique):,} unique tickers via yfinance ...")
    cache: dict[str, dict] = {}
    for i, ticker in enumerate(unique, 1):
        try:
            info = yf.Ticker(ticker).info
            has_price = bool(
                info.get('regularMarketPrice')
                or info.get('currentPrice')
                or info.get('previousClose')
            )
            cache[ticker] = {
                'yf_name':     info.get('shortName') or info.get('longName', ''),
                'yf_exchange': _EXC_MAP.get(info.get('exchange', ''),
                                            info.get('exchange', '')),
                'valid':       has_price,
            }
        except Exception:
            cache[ticker] = {'yf_name': '', 'yf_exchange': '', 'valid': False}
        if i % 25 == 0:
            print(f"    {i}/{len(unique)}")
        time.sleep(0.25)
    return cache

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # 1. Compute petition counts from all LCA data
    petition_counts = compute_petitions()
    employers = {n: c for n, c in petition_counts.items() if c >= MIN_PETITIONS}
    print(f"  {len(employers):,} employers with >= {MIN_PETITIONS} petitions\n")

    # 2. Load previously confirmed matches (skip not_found so they get re-tried)
    lookup = load_existing_lookup()

    # 3. Separate employers into: already resolved vs needs matching
    resolved_rows: list[dict] = []
    queue: list[tuple[str, int]] = []   # (EMPLOYER_NORM, petitions) to process

    for name, petitions in sorted(employers.items(), key=lambda x: -x[1]):
        if name in lookup:
            m = lookup[name]
            resolved_rows.append({
                'EMPLOYER_NORM':   name,
                'TICKER':          m['ticker'],
                'EXCHANGE':        m['exchange'],
                'LISTED_NAME':     m['listed_name'],
                'MATCH_TYPE':      m['match_type'],
                'MATCH_SCORE':     m['match_score'],
                'TOTAL_PETITIONS': petitions,
            })
        else:
            queue.append((name, petitions))

    print(f"  {len(resolved_rows):,} reusing existing match, "
          f"{len(queue):,} to process\n")

    # 4. Load NASDAQ listing
    nasdaq_names, nasdaq_symbols = [], []
    if NASDAQ_FILE.exists():
        nasdaq_names, nasdaq_symbols = load_nasdaq_listings(NASDAQ_FILE)
        print(f"Loaded {len(nasdaq_symbols):,} NASDAQ equities from {NASDAQ_FILE.name}\n")
    else:
        print(f"[warn] {NASDAQ_FILE} not found — NASDAQ matching skipped\n")

    # 5. Match the queue: manual → NASDAQ → yfinance search
    new_rows: list[dict] = []
    tickers_to_validate: list[str] = []

    print(f"Matching {len(queue):,} employers ...")
    for i, (name, petitions) in enumerate(queue, 1):
        ticker  = None
        exchange, listed_name, score = '', '', 0

        # Step 5a: manual patterns
        ticker, _ = apply_manual(name)
        if ticker is not None:
            match_type = 'not_traded' if ticker == '' else 'manual'
            score      = 100 if ticker else 0
        else:
            ticker = ''
            # Step 5b: NASDAQ fuzzy match
            if nasdaq_names:
                threshold = FUZZY_MIN if petitions >= YF_MIN_PETITIONS else FUZZY_MIN_SMALL
                ticker, exchange, score = nasdaq_match(
                    normalize(name), nasdaq_names, nasdaq_symbols, threshold)
                match_type = 'listing' if ticker else 'not_found'

            # Step 5c: yfinance search (large employers only)
            if not ticker and petitions >= YF_MIN_PETITIONS:
                ticker, exchange, score = yf_search(name)
                match_type = 'search' if ticker else 'not_found'
                time.sleep(0.25)

            if not ticker:
                match_type = 'not_found'

        if ticker:
            tickers_to_validate.append(ticker)

        new_rows.append({
            'EMPLOYER_NORM':   name,
            'TICKER':          ticker,
            'EXCHANGE':        exchange,
            'LISTED_NAME':     listed_name,
            'MATCH_TYPE':      match_type,
            'MATCH_SCORE':     score,
            'TOTAL_PETITIONS': petitions,
        })
        if i % 5000 == 0:
            print(f"  {i:,}/{len(queue):,}")

    # 6. Validate newly matched tickers via yfinance
    yf_cache: dict[str, dict] = {}
    if tickers_to_validate:
        print()
        yf_cache = validate_tickers(tickers_to_validate)

    def enrich(row: pd.Series) -> pd.Series:
        t = str(row['TICKER']).strip()
        if not t:
            return row
        d = yf_cache.get(t, {})
        if not d.get('valid'):
            if row['MATCH_TYPE'] in ('listing', 'search'):
                row['TICKER']     = ''
                row['EXCHANGE']   = ''
                row['MATCH_TYPE'] = 'not_found'
                row['MATCH_SCORE'] = 0
        else:
            if not str(row['LISTED_NAME']).strip():
                row['LISTED_NAME'] = d.get('yf_name', '')
            if not str(row['EXCHANGE']).strip() or row['MATCH_TYPE'] == 'manual':
                row['EXCHANGE'] = d.get('yf_exchange', row['EXCHANGE'])
        return row

    new_df = pd.DataFrame(new_rows).apply(enrich, axis=1)

    # 7. Combine and write
    out = pd.concat([pd.DataFrame(resolved_rows), new_df], ignore_index=True)
    out = out.sort_values('TOTAL_PETITIONS', ascending=False).reset_index(drop=True)
    out.to_csv(OUT_FILE, index=False, encoding='utf-8')

    # Summary
    matched    = out[out['TICKER'].fillna('') != '']
    not_traded = out[out['MATCH_TYPE'] == 'not_traded']
    not_found  = out[out['MATCH_TYPE'] == 'not_found']

    print(f"\n{'='*60}")
    print(f"ticker_mapping.csv  ({len(out):,} employers, >= {MIN_PETITIONS} petitions)")
    print(f"  Has ticker (matched) : {len(matched):>7,}  "
          f"({len(matched)/len(out)*100:.1f}%)")
    print(f"    manual             : {(out['MATCH_TYPE']=='manual').sum():>7,}")
    print(f"    NASDAQ listing     : {(out['MATCH_TYPE']=='listing').sum():>7,}")
    print(f"    yfinance search    : {(out['MATCH_TYPE']=='search').sum():>7,}")
    print(f"  Not traded (confirmed): {len(not_traded):>6,}")
    print(f"  Not found            : {len(not_found):>7,}")
    print(f"  Unique tickers       : {out['TICKER'].replace('', pd.NA).nunique():>7,}")
    total_pet = out['TOTAL_PETITIONS'].astype(int).sum()
    match_pet = matched['TOTAL_PETITIONS'].astype(int).sum()
    print(f"  Petition coverage    : {match_pet/total_pet*100:.1f}%"
          f"  ({match_pet:,} / {total_pet:,})")
    print(f"\nWrote {len(out):,} rows → {OUT_FILE}")

    print("\nTop 20 employers by petition count:")
    preview_cols = ['EMPLOYER_NORM', 'TICKER', 'EXCHANGE', 'MATCH_TYPE', 'TOTAL_PETITIONS']
    print(out[preview_cols].head(20).to_string(index=False))


if __name__ == '__main__':
    main()
