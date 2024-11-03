import os
import re
from typing import List, Optional
from urllib.parse import quote

import requests
from pymongo import MongoClient
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()
# Define the Pydantic models


class Paragraf(BaseModel):
    cislo: str
    zneni: str


class Law(BaseModel):
    nazev: str
    id: str
    year: str
    category: Optional[str]
    date: Optional[str]
    staleURL: Optional[str] = None
    paragrafy: List[Paragraf] = []

# Function to parse the text into Law instances


def parse_laws(text: str) -> List[Law]:
    laws = []
    current_category = None
    category_list = [
        'FINANCE',
        'MEZINÁRODNÍ PRÁVO',
        'OBČANSKÉ PRÁVO',
        'Obchod a podnikání',
        'PRACOVNÍ PRÁVO',
        'SPRÁVNÍ PRÁVO',
        'TRESTNÍ PRÁVO',
        'ÚSTAVNÍ PRÁVO',
        'HOSPODÁŘSTVÍ'
    ]
    lines = text.strip().split('\n')
    law_pattern = re.compile(
        r'^(\d+)/(\d+)\s+Sb[.]?\s+(.*?)(?:\s+(\d{2}[.]\d{2}[.]\d{4}))?$'
    )
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Ignore column headers
        if line.startswith('Číslo'):
            continue
        # Check if line is a category header
        if line in category_list:
            current_category = line
            continue
        # Check if line is a law line
        match = law_pattern.match(line)
        if match:
            law_id, year, name, date = match.groups()
            law = Law(
                nazev=name.strip(),
                id=law_id.strip(),
                year=year.strip(),
                date=date.strip() if date else None,
                category=current_category,
            )
            laws.append(law)
    return laws

# Function to get MongoDB client


def get_mongo_client():
    client = MongoClient("mongodb://localhost:27017/")
    return client

# Function to save Law instance to MongoDB


def save_law_to_mongodb(law: Law, db_name='law_database_mvp', collection_name='MVP'):
    client = get_mongo_client()
    db = client[db_name]
    collection = db[collection_name]
    law_doc = law.dict()
    collection.insert_one(law_doc)
    print(f"Law {law.year}/{law.id} saved to collection '{collection_name}'.")

# Function to fetch law details and paragraphs from the API


def get_law_details(law: Law, api_key: str) -> Law:
    headers = {"esel-api-access-key": api_key}
    # Construct the 'sign'
    sign = f"/sb/{law.year}/{law.id}/0000-00-00"
    sign_encoded = quote(sign, safe='')

    base_url = "https://api.e-sbirka.cz/dokumenty-sbirky/"
    # Get the law metadata
    url = base_url + sign_encoded
    try:
        res = requests.get(url, headers=headers)
        res.raise_for_status()
        data = res.json()
    except Exception as e:
        print(
            f"Failed to fetch law details for {law.year}/{law.id}. Error: {e}")
        return law

    # Check if law is cancelled
    datum_zruseni = data.get('datumZruseni')
    if datum_zruseni:
        from datetime import datetime
        cancellation_date = datetime.strptime(datum_zruseni, '%Y-%m-%d')
        if cancellation_date < datetime.now():
            raise ValueError(
                f"Law {law.year}/{law.id} was cancelled on {datum_zruseni}")

    law.datumZruseni = datum_zruseni
    law.nazev = data.get('nazev')
    law.staleURL = data.get('staleUrl')

    # Now fetch the fragments (paragraphs)
    page_number = 0
    num_pattern = r'\d+'
    text_pattern = r'<[^>]+>'
    while True:
        fragments_url = f"{base_url}{sign_encoded}/fragmenty?cisloStranky={page_number}"
        try:
            print(f"Fetching page {page_number} for law {law.year}/{law.id}")
            resp = requests.get(fragments_url, headers=headers)
            resp.raise_for_status()
            response = resp.json()
        except Exception as e:
            print(
                f"Failed to fetch fragments for law {law.year}/{law.id}, page {page_number}. Error: {e}")
            break

        fragments = response.get('seznam', [])
        if not fragments:
            # If no fragments are returned, break the loop
            print(
                f"No fragments found on page {page_number} for law {law.year}/{law.id}")
            break

        current_paragraph_number = None
        current_paragraph_text = ""
        try:
            for fragment in fragments:
                kod_typu_fragmentu = fragment.get('kodTypuFragmentu')
                xhtml_content = fragment.get('xhtml', '')

                if kod_typu_fragmentu == 'Paragraf':
                    if current_paragraph_number is not None:
                        # Save the current paragraph before starting a new one
                        law.paragrafy.append(Paragraf(
                            cislo=current_paragraph_number,
                            zneni=current_paragraph_text.strip()
                        ))
                    # Start a new paragraph
                    match = re.search(num_pattern, xhtml_content)
                    if match is None:
                        print(
                            f"Paragraf number not found in fragment for law {law.year}/{law.id}")
                        continue
                    current_paragraph_number = match.group()
                    current_paragraph_text = ""
                else:
                    # Append fragment text to the current paragraph
                    cleaned_text = re.sub(text_pattern, '', xhtml_content)
                    current_paragraph_text += cleaned_text + "\n"

            # Save the last paragraph if any
            if current_paragraph_number is not None and current_paragraph_text.strip():
                law.paragrafy.append(Paragraf(
                    cislo=current_paragraph_number,
                    zneni=current_paragraph_text.strip()
                ))
        except Exception as e:
            print(
                f"Error processing fragments for law {law.year}/{law.id}, page {page_number}. Error: {e}")

        page_number += 1  # Increment page number to fetch the next page

    return law


def main():
    # Read the text of laws from the given variable
    text = '''

HOSPODÁŘSTVÍ
441/2003 Sb.	Zákon o ochranných známkách	01.04.2004
143/2001 Sb.	Zákon o ochraně hospodářské soutěže	01.07.2001
134/2016 Sb.	Zákon o zadávání veřejných zakázek (nový)	01.10.2016
477/2001 Sb.	Zákon o obalech	01.01.2002


FINANCE

120/2001 Sb.	Exekuční řád	01.05.2001
586/1992 Sb.	Zákon o daních z příjmů	01.01.1993
235/2004 Sb.	Zákon o dani z přidané hodnoty	01.05.2004
280/2009 Sb.	Daňový řád	01.01.2011
563/1991 Sb.	Zákon o účetnictví	01.01.1992
549/1991 Sb.	Zákon o soudních poplatcích	01.01.1992
235/2004 Sb    Zákon o dani z přidané hodnoty
127/2005 Sb.	Zákon o elektronických komunikacích	01.05.2005
256/2004 Sb.   Zákon o podnikání na kapitálovém trhu


MEZINÁRODNÍ PRÁVO
326/1999 Sb.	Zákon o pobytu cizinců na území ČR	01.01.2000
209/1992 Sb.	Úmluva o ochraně lidských práv a základních svobod a Protokoly k ní	03.09.1953
325/1999 Sb.	Zákon o azylu	01.01.2000
104/1991 Sb.	Úmluva o právech dítěte	06.02.1991
38/1994 Sb.	Zákon o zahraničním obchodu s vojenským materiálem	01.04.1994
160/1991 Sb.	Úmluva OSN o smlouvách o mezinárodní koupi zboží	01.04.1991
15/1988 Sb.	Vídeňská úmluva o smluvním právu	27.01.1980


OBČANSKÉ PRÁVO

Číslo	Název předpisu	Účinnost od
89/2012 Sb.	Občanský zákoník (nový)	01.01.2014
99/1963 Sb.	Občanský soudní řád	01.04.1964
182/2006 Sb.	Insolvenční zákon	01.01.2008
292/2013 Sb.	Zákon o zvláštních řízeních soudních	01.01.2014
253/2008 Sb.	Zákon o některých opatřeních proti legalizaci výnosů z trestné činnosti a financování terorismu
634/1992 Sb.	Zákon o ochraně spotřebitele	31.12.1992
159/1999 Sb.	Zákon o některých podmínkách podnikání v oblasti cestovního ruchu	01.10.2000
359/1999 Sb.	Zákon o sociálně-právní ochraně dětí	01.04.2000
256/2013 Sb.    Katastrální zákon	01.01.2014
357/2013 Sb.	Vyhláška o katastru nemovitostí (katastrální vyhláška)	01.01.2014
499/2004 Sb.	Zákon o archivnictví a spisové službě	01.01.2005
121/2000 Sb.	Autorský zákon	01.12.2000
37/2021 Sb.	Zákon o evidenci skutečných majitelů	01.06.2021
301/2000 Sb.	Zákon o matrikách, jménu a příjmení	01.07.2001
526/1990 Sb.	Zákon o cenách	01.01.1991

Obchod a podnikání
90/2012 Sb.	Zákon o obchodních korporacích	01.01.2014
455/1991 Sb.	Živnostenský zákon	01.01.1992
64/1986 Sb.	Zákon o České obchodní inspekci	01.01.1987


PRACOVNÍ PRÁVO
262/2006 Sb.	Zákoník práce	01.01.2007
108/2006 Sb.	Zákon o sociálních službách	01.01.2007
177/1996 Sb.	Advokátní tarif	01.07.1996
258/2000 Sb.	Zákon o ochraně veřejného zdraví	01.01.2001
435/2004 Sb.	Zákon o zaměstnanosti	01.10.2004
361/2003 Sb.	Zákon o služebním poměru příslušníků bezpečnostních sborů	01.01.2007
234/2014 Sb.	Zákon o státní službě	01.01.2015
117/1995 Sb.	Zákon o státní sociální podpoře	01.10.1995
187/2006 Sb.	Zákon o nemocenském pojištění	01.01.2009
155/1995 Sb.	Zákon o důchodovém pojištění	01.01.1996
111/2006 Sb.	Zákon o pomoci v hmotné nouzi	01.01.2007

SPRÁVNÍ PRÁVO
Číslo	Název předpisu	Účinnost od
541/2020 Sb.   Zákon o odpadech 	01.01.2021
283/2021 Sb.	Stavební zákon (nový)	01.01.2024
361/2000 Sb.	Zákon o silničním provozu	01.01.2001
500/2004 Sb.	Správní řád	01.01.2006
561/2004 Sb.	Školský zákon	01.01.2005
128/2000 Sb. Zákon o obcích
254/2001 Sb.	Vodní zákon	01.01.2002
250/2016 Sb.	Zákon o odpovědnosti za přestupky a řízení o nich	01.07.2017
251/2016 Sb.	Zákon o některých přestupcích	01.07.2017
13/1997 Sb.	Zákon o pozemních komunikacích	01.04.1997)
458/2000 Sb. energetický zákon
372/2011 Sb.	Zákon o zdravotních službách	01.04.2012
106/1999 Sb.	Zákon o svobodném přístupu k informacím	01.01.2000


TRESTNÍ PRÁVO
40/2009 Sb.	Trestní zákoník	01.01.2010
141/1961 Sb.	Trestní řád	01.01.1962
218/2003 Sb.	Zákon o soudnictví ve věcech mládeže	01.01.2004
171/2023 Sb.	Zákon o ochraně oznamovatelů	01.08.2023
418/2011 Sb.	Zákon o trestní odpovědnosti právnických osob a řízení proti nim	01.01.2012
45/2013 Sb.	Zákon o obětech trestných činů	01.08.2013

ÚSTAVNÍ PRÁVO
273/2008 Sb.	Zákon o Policii ČR	01.01.2009
2/1993 Sb.	Listina základních práv a svobod	01.01.1993
240/2000 Sb.	Krizový zákon	01.01.2001
ÚSTAVA ČR
127/2005 Sb.	Zákon o elektronických komunikacích	01.05.2005
150/2002 Sb.	Soudní řád správní	01.01.2003
6/2002 Sb.	Zákon o soudech a soudcích	01.04.2002

    '''

    # Parse the laws
    laws = parse_laws(text)

    # Get API key
    open_data_api_key = os.getenv("OPEN_DATA_API_KEY")
    if not open_data_api_key:
        raise ValueError("OPEN_DATA_API environment variable not set")
    fetched_laws = []
    # Process each law
    for law in laws:
        print(f"Processing law {law.year}/{law.id}")
        law = get_law_details(law, open_data_api_key)
        save_law_to_mongodb(law)


if __name__ == '__main__':
    main()
