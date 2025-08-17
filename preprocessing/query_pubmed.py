import re
import json
import pandas as pd
import unicodedata
from datetime import datetime
import os
import random
import requests
import time
from bs4 import BeautifulSoup


QUERY_STRING = '("Type 1 Diabetes"[Title/Abstract] OR "T1D"[Title/Abstract]) AND ("Pediatric"[Title/Abstract] OR "Pediatrics"[Title/Abstract] OR "Neonate"[Title/Abstract] OR "Neonates"[Title/Abstract] OR "Infant"[Title/Abstract] OR "Infants"[Title/Abstract] OR "Children"[Title/Abstract] OR "Adolescent"[Title/Abstract] OR "Adolescents"[Title/Abstract]) AND ("open access"[filter])'



class QueryNCBIManager():
    def __init__(self,
                 query,
                 starting_year,
                 ending_year,
                 database_system,
                 output_base_dir,
                 max_results=10000,
                 ):
        self.query = query
        self.max_results = max_results
        self.database_system = database_system

        if ending_year is None:
            ending_year = datetime.today().year
        if ending_year < starting_year:
            raise ValueError("Ending year for search should be greater than starting year.")
        
        self.starting_year = starting_year
        self.ending_year = ending_year

        self.output_base_dir = output_base_dir
        
        os.makedirs(os.path.join(self.output_base_dir), exist_ok=True)  

        self.ncbi_ids = {y: [] for y in range(self.starting_year, self.ending_year + 1)}
        self.files = {y: [] for y in range(self.starting_year, self.ending_year + 1)}
        

    def search(self):
        for year in self.ncbi_ids.keys():
            print(f"\nYear: {year}")

            os.makedirs(os.path.join(self.output_base_dir, self.database_system, str(year)), exist_ok=True)  
            self.ncbi_ids[year] = self._search_per_year(year)

            print(f"Number of articles returned in the query (DB: {self.database_system}): {len(self.ncbi_ids[year])}")
            self.save_raw_xml_files(year)


    def _pause_query(self):
        # Pause the processing for 1 to 2 seconds
        pause_duration = random.uniform(1, 2)
        time.sleep(pause_duration)


    def _search_per_year(self, year):
        """
        Method creating a new attribute with the Unique IDs (UID) of each of the articles retrieved 
        https://www.ncbi.nlm.nih.gov/books/NBK25501/ 
        We must be careful they don't block our IP address. NCBI recommends that users post 
        no more than three URL requests per second and limit large jobs to either weekends or 
        between 9:00 PM and 5:00 AM Eastern time during weekdays. 
        We can go beyond the three URL requests per second by requesting an API key.
        https://www.ncbi.nlm.nih.gov/books/NBK25497/

        Parameters
        : year (int) : year related to the query
        """
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"

        # Parameters can be found here: https://dataguide.nlm.nih.gov/eutilities/utilities.html
        params = {
            "db": self.database_system,
            "term": f"{self.query} AND ({year}[pdat])",
            "retmax": self.max_results,
            "retmode": "xml"
        }
        response = requests.get(base_url, params=params)  # Perform an HTML get request
        self._pause_query()  # Pauses the querying for 1-2 seconds to prevent our IP address from being blocked by NCBI

        if response.status_code == 200:  # Status 200 = OK
            soup = BeautifulSoup(response.content, features='lxml')  # Use the BeautifulSoup constructor to parse the XML content
            ids = [id_tag.text for id_tag in soup.find_all("id")]  # Retrieve a list of XML 'id' elements. These are UIDs
            return ids
        else:
            print(f"Error: Unable to fetch results from database: {self.database_system}")
            return []
        

    def _fetch_pmc_article(self, pmc_id):
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        params = {
            "db": self.database_system,
            "id": pmc_id, # where is this coming from 
            "retmode": "xml",
            'rettype': "full"
        }
        # Retrieve (fetch) each article based on the Unique Identification (UID). UID likely the primary key. 
        response = requests.get(base_url, params=params)

        # Pauses the querying for 1-2 seconds to prevent our IP address from being blocked by NCBI
        self._pause_query()
        if response.status_code == 200:
            return response.content
        else:
            print(f"Error: Unable to fetch article for PMC ID {pmc_id}")
            return None
            
        
    def _check_existing_file(self, file_name, year):
        """
        Check if the file already exists in either the 'pmc' or 'pubmed' directories.

        Parameters:
        - raw_xml_base_directory: The base directory where the files are saved.
        - file_name: The name of the file to check.
        - db: The current database ('pmc' or 'pubmed').
        - year: The year of the folder to check.

        Returns:
        - True if the file exists in either directory, False otherwise.
        """
        pmc_dir = f"{self.output_base_dir}/pmc/{year}"
        pubmed_dir = f"{self.output_base_dir}/pubmed/{year}"

        pmc_path = os.path.join(pmc_dir, file_name)
        pubmed_path = os.path.join(pubmed_dir, file_name)

        if self.database_system == 'pmc':
            # check if file exists in the pmc folder because we retrieve from pmc folder even if it's in the pubmed folder # TODO 
            return os.path.exists(pmc_path)
        else: # pubmed
            return os.path.exists(pmc_path) or os.path.exists(pubmed_path)
            
    
    def save_raw_xml_files(self, year): # output_dir
        """ 
        This function saves pmc_ids files into xml files which can retrieve in raw_xml_base_directory/db/year.
        The function pause for 1-2 seconds before fetching each  article. NCBI can block an IP address if there are 
        more than three queries per second. 
        Parameters
        : pmc_ids : list of ids for pubmed publications as retrieved by the search_NCBI_database function. 
        : raw_xml_base_directory : the base directory where the files will be saved
        : db : the database to search, either 'pmc' or 'pubmed'
        : year : the year of the publication
        
        """

        pmc_id_length = len(self.ncbi_ids[year])
        if pmc_id_length == 0:
            return
        
        files = []
        for index, pmc_id in enumerate(self.ncbi_ids[year]):  # loop through the pmc_ids
            filename = f"{pmc_id}.xml"

            if self._check_existing_file(filename, year):  # check if the file already exists in both directories there is no need to retrieve it again
                continue

            xml_file_path_name = os.path.join(f"{self.output_base_dir}/{self.database_system}/{year}", filename)
            
            article = self._fetch_pmc_article(pmc_id)  # fetch and save a single article
            if article:
                with open(xml_file_path_name, "wb") as f:
                    f.write(article)
                f.close
                files.append(filename)

            if ((index+1) < pmc_id_length):
                print(f"\rProgress retrieving the articles: {round(((index+1)/pmc_id_length)*100, 2)}%", end="")
            else:
                print(f"\rProgress retrieving the articles: {round(((index+1)/pmc_id_length)*100, 2)}%")

        self.files[year] = files


class XMLConverter():
    def __init__(self,
                 filedict,
                 database_system,
                 output_base_dir,
                 output_base_extr,
                 output_json
                 ):
        self.filedict = filedict
        self.database_system = database_system
        self.output_base_dir = output_base_dir
        self.output_base_extr = output_base_extr
        self.output_json = output_json

        os.makedirs(os.path.join(self.output_base_extr), exist_ok=True)  

    def save_metadata_cleaned_xmls(self):
        for year in self.filedict.keys():
            print(f"\nYear: {year}")
            path_extr_db_year = os.path.join(self.output_base_extr, self.database_system, str(year))
            path_base_db_year = os.path.join(self.output_base_dir, self.database_system, str(year))
            os.makedirs(path_extr_db_year, exist_ok=True)  
            self.save_dataset_to_metadata_csv(path_extr_db_year, path_base_db_year, year)

            self.save_dataset_to_json(path_extr_db_year, path_base_db_year, year)


    def _extract_article_metadata(self, soup, year):
        # Extract article identifiers
        id_article = {"pmc": "article-id", "pubmed": "ArticleID"}
        id_type = {"pmc": "pub-id-type", "pubmed": "IdType"}
        id_title = {"pmc": "article-title", "pubmed": "ArticleTitle"}
        
        pmc = soup.find(id_article[self.database_system], {id_type[self.database_system]: "pmc"})
        doi = soup.find(id_article[self.database_system], {id_type[self.database_system]: "doi"})
        article_title = "N/A"
        article_meta = True

        license_type = 'Unknown License Type'
        license_url = 'No License URL Found'
        copyright_text = "No Copyright Information Found"

        if self.database_system == 'pmc':
            pmid = soup.find("article-id", {"pub-id-type": "pmid"})
            article_meta = soup.find("article-meta")

            license_info = soup.find('license')
            if license_info:
                license_type = license_info.get('license-type', 'Unknown License Type')
                ext_link = soup.find('ext-link', {'ext-link-type': 'uri'})
                license_url = ext_link.get('xlink:href', 'No License URL Found') if ext_link else "No License URL Found"
                copyright_tag = soup.find('copyright-statement')
                copyright_text = copyright_tag.get_text(strip=True) if copyright_tag else "No Copyright Information Found"

        else:
            copyright_info = soup.find('CopyrightInformation')
            if copyright_info:
                copyright_text = copyright_info.get_text() if copyright_info else "No Copyright Information Found"
            pmid = soup.find("PMID")

        if article_meta:
            article_title_tag = soup.find(id_title[self.database_system])
            article_title = article_title_tag.text.strip() if article_title_tag else "N/A"

        pmid = pmid.text.strip() if pmid else "N/A" # extract content of set N/A if missing
        pmc = pmc.text.strip() if pmc else "N/A"
        doi = doi.text.strip() if doi else "N/A"

        return {
            "pmid": pmid,
            "pmc": pmc,
            "doi": doi,
            "article-title": article_title,
            "license-type": license_type,
            "license-url": license_url,
            "copyright-text": copyright_text,
            "year": year,
            }
    

    def _from_corpus_to_sections(self, sec):
        if self.database_system == 'pmc':
            sec_id = sec.get("id", "N/A")  # get the 'id' attribute, or 'N/A' if missing
            title = sec.find("title")
            title_text = title.text.strip() if title else "N/A"  # Get the title text, or 'N/A'
            return {"sec_id": sec_id, "title": title_text}
        else:
            label = sec.get("Label", "N/A")  # Extract the 'Label' attribute if it exists
            nlm_category = sec.get("NlmCategory", "N/A")  # Extract the 'NlmCategory' attribute if it exists
            return {"label": label, "nlm_category": nlm_category}


    def extract_metadata_and_sections(self, file_path, year):
        # try:
        abstract_field = {"pmc": "abstract", "pubmed": "Abstract"}
        abstract_txt = {"pmc": "sec", "pubmed": "AbstractText"}

        with open(file_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "xml")
        # except:
        #    print(f"File not found: {}")
        metadata = self._extract_article_metadata(soup, year)
        abstract = soup.find(abstract_field[self.database_system]) #Abstract

        abstract_sections = []
        if abstract:
            for sec in abstract.find_all(abstract_txt[self.database_system]): #AbstractText
                abstract_sections.append(self._from_corpus_to_sections(sec))

        if self.database_system == 'pubmed':
            body = False
            body_sections = None
        else:
            body = soup.find("body")  # this part of file will be found independently from additional attributes
            body_sections = []  # list to store body sections
            if body:
                for sec in body.find_all("sec"):  # extract all <sec> elements within the <body>
                    body_sections.append(self._from_corpus_to_sections(sec))

                # extract all <p> elements that are not within <sec> tags
                for p in body.find_all("p", recursive=False):  # Only consider direct children of <body>
                    # Assign 'blank' to sec_id and title for <p> tags outside <sec> tags
                    body_sections.append({"sec_id": "blank", "title": "blank"})

        return {
            "pmid": metadata['pmid'],
            "pmc": metadata['pmc'],
            "doi": metadata['doi'],
            "article-title": metadata['article-title'],
            "year": metadata['year'],
            "license-type": metadata["license-type"],
            "license-url": metadata["license-url"],
            "copyright-text": metadata["copyright-text"],
            "abstract-exists": bool(abstract),
            "abstract-sections": abstract_sections,
            "body-exists": bool(body),
            "body-sections": body_sections
        }


    def save_dataset_to_metadata_csv(self, path_to_file, path_to_output, year):  # path to file is output_dir_of_extracted_files, path to output = output_dir 
        col_names = ["pmid", "pmc", "doi", "article-title", "year", "license-type", "license-url", "copyright-text",
                      "abstract-exists", "abstract-sections", "body-exists", "body-sections"] 
    
        path_to_all = os.path.join(path_to_file, f"_all_file_metadata.csv")
        if os.path.exists(path_to_all):
            df_all_file = pd.read_csv(path_to_all, index_col=0)
        else:
            df_all_file = pd.DataFrame(columns=col_names)

        path_to_invalid = os.path.join(path_to_file, f"_invalid_file_metadata.csv")
        if os.path.exists(path_to_invalid):
            df_invalid_file = pd.read_csv(path_to_invalid, index_col=0)
        else:
            df_invalid_file = pd.DataFrame(columns=col_names)
 
        if self.filedict[year] is None or len(self.filedict[year]) == 0:
            return
        
        info_available = []
        info_n_available = []

        for index, file in enumerate(self.filedict[year]):
            article_metadata = self.extract_metadata_and_sections(os.path.join(path_to_output, file), year)
            if article_metadata is not None:
                if article_metadata['body-exists']:
                    article_metadata["body-sections"] = {section['sec_id']: section['title'] for section in article_metadata["body-sections"]}
                    # Write the extract files into a new directory for those extracted files
                    info_available.append(article_metadata)
                else:
                    article_metadata["body-sections"] = {}
                    info_n_available.append(article_metadata)
            
            if ((index+1) < len(self.filedict[year])):
                print(f"\rProgress processing the metadata: {round(((index+1)/len(self.filedict[year]))*100, 2)}%", end="")
            else:
                print(f"\rProgress processing the metadata: {round(((index+1)/len(self.filedict[year]))*100, 2)}%")
            
        tmp_df_all = pd.DataFrame(data=info_available, columns=col_names)
        tmp_df_invalid = pd.DataFrame(data=info_n_available,  columns=col_names)
        
        combined_df_all = pd.concat([df_all_file, tmp_df_all], ignore_index=True)
        combined_df_invalid = pd.concat([df_invalid_file, tmp_df_invalid], ignore_index=True)
        
        combined_df_all.to_csv(os.path.join(path_to_file, f"_all_file_metadata.csv"))
        combined_df_invalid.to_csv(os.path.join(path_to_file, f"_invalid_file_metadata.csv"))

    def preprocess_text(self, text):
        """
        This code is for data preprocessing. Convert the input text to its canonical decomposition form. 
        The normalization form "NFKD" means Normalization Form KC (Compatibility Decomposition). 
        It helps in standardizing the text by removing diacritical marks and other unicode-specific variations.
        """
        text = unicodedata.normalize("NFKD", text)
        
        # This line removes citation references, which are typically in the format `[number]` or `[number, number, ...]`.
        # The `re.sub` function uses a regular expression pattern to identify and remove these citations from the text.
        ### This is handled in the clean_xref_tags function below
        # text = re.sub(r"\s*\[\d+(?:,\s*\d+)*\]", "", text)
        text = re.sub(r"\s+", " ", text).strip() # replacing multiple consecutive whitespace characters (spaces, tabs, newlines) with a single space
        return text
    

    def clean_xref_tags(self, xml_content):
        """
        Remove the reference numbers and surrounding [] and () from the text.
        Cleans <xref> tags, surrounding brackets [] and (), and separators like dashes or commas from the XML content.
        The <xref> elements contain the reference numbers within the XML file.
        """
        xml_content = re.sub(  #remove patterns like [ <xref>1</xref> – <xref>3</xref> ] and ( <xref>1</xref> – <xref>3</xref> )
            r"\s*[\[\(]?\s*<xref[^>]*>.*?</xref>\s*(?:[-–,]\s*<xref[^>]*>.*?</xref>)*\s*[\]\)]?",
            "",
            xml_content,
        )
        xml_content = re.sub(r"<xref[^>]*>.*?</xref>", "", xml_content)  # remove standalone <xref> tags if any remain
        return xml_content
    

    def _extract_sections_within_portion(self, chunk, abstract=True): # it can either be from the abstract of the main body
        fields = ['sec', 'p']
        if abstract:
            fields.append('AbstractText')

        sequence = chunk.find_all(['sec', 'p', 'AbstractText'], recursive=False)  # process <sec> and <p> tags
        
        content = {}
        previous_section_title = "Introduction"  # default for the first section if no title exists
        for element in sequence:  # process <sec> and <p> tags
            report_sec = True
            if element.name == 'sec':
                title_element = element.find("title")
                # use the previous section title if no title exists, otherwise use the current title
                section_title = title_element.text.strip() if title_element and title_element.text.strip() else previous_section_title
                if title_element:
                    title_element.decompose()  # remove the title tag to avoid duplication
                previous_section_title = section_title  # update the previous section title

            elif element.name == 'p' or element.name == 'AbstractText':  # new paragraph or extract text directly from AbstractText element
                section_title = previous_section_title  # inherit the previous section title
            else:
                report_sec = False

            if report_sec: 
                section_content = element.get_text(separator=" ", strip=True)
                content[section_title] = (content.get(section_title, "") + " " + self.preprocess_text(section_content)).strip()
        return content


    def extract_sections_within_xml(self, xml_file):
        with open(xml_file, "r", encoding="utf-8") as f:  # read the XML file content
            xml_content = f.read()

        # clean the XML content by removing unwanted <xref> tags and structures before converting the XML using BeautifulSoup
        cleaned_xml_content = self.clean_xref_tags(xml_content)
        soup = BeautifulSoup(cleaned_xml_content, "xml")  # parse the cleaned XML with BeautifulSoup

        # We need to check if any <xref> remain calling the clean_xref_tags function above
        #for xref in soup.find_all('xref'):
        #    xref.decompose()

        for table_wrap in soup.find_all('table-wrap'):  # removes all tables within <table-wrap> elements
            table_wrap.decompose()

        for fig in soup.find_all('fig'):  # remove all figures which within the <fig> elements
            fig.decompose()

        metadata = self._extract_article_metadata(soup, year=None)

        abstract = soup.find("Abstract") or soup.find("abstract") or soup.find("AbstractText")  # as named by pubmed or pmc
        if abstract:
            abstract_content = self._extract_sections_within_portion(abstract)
        else:
            abstract_content = {}
            
        body = soup.find("body")  # body of the doc
        if body:
            body_content = self._extract_sections_within_portion(body, abstract=False)
        else:
            body_content = {}

        license_type = 'Unknown License Type'
        license_url = 'No License URL Found'
        copyright_text = "No Copyright Information Found"
        if self.database_system == 'pmc':
            license_info = soup.find("license")
            if license_info:
                license_type = license_info.get('license-type', 'Unknown License Type')
                ext_link = soup.find('ext-link', {'ext-link-type': 'uri'})
                license_url = ext_link.get('xlink:href', 'No License URL Found') if ext_link else "No License URL Found"
                copyright_info = soup.find('copyright-statement')
                copyright_text = copyright_info.get_text() if copyright_info else "No Copyright Information Found"                
        else:
            copyright_info = soup.find('CopyrightInformation')
            if copyright_info:
                copyright_text = copyright_info.get_text() if copyright_info else "No Copyright Information Found"

        # Extract license type and URL
         # PMC uses the 'pmc' key, while PubMed uses the 'pmid' key.
        key = metadata[{'pmc': 'pmc', 'pubmed': 'pmid'}[self.database_system]]
        
        # extracted_content = {}
        # Build the output dictionary
        # extracted_content[key] = {
        
        extracted_content = {
            "pmid": metadata['pmid'],
            "pmc": metadata['pmc'],
            "doi": metadata['doi'],
            "article-title": metadata['article-title'],
            "license-type": license_type,
            "license-url": license_url,
            "copyright-text": copyright_text,
            "abstract-content": abstract_content,
            "body-content": body_content,
        }
        
        return key, extracted_content


    def save_dataset_to_json(self, path_extr_db_year, path_base_db_year, year):

        if self.filedict[year] is None: 
            return
        files_len = len(self.filedict[year])
        if files_len == 0:
            return

        try: 
            with open(os.path.join(path_extr_db_year, self.output_json), 'r', encoding='utf-8') as f:
                data = json.load(f)
                if not isinstance(data, dict):
                    raise ValueError("JSON file does not contain a dictionary at the top level.")
        except:
            print("New JSON file")
            with open(os.path.join(path_extr_db_year, self.output_json), 'w', encoding="utf-8") as f:
                data = {}
        
        for index, file in enumerate(self.filedict[year]): 
            key, content_entry = self.extract_sections_within_xml(os.path.join(path_base_db_year, file))
            data[key] = content_entry  # add new key-value pair
            with open(os.path.join(path_extr_db_year, self.output_json), 'w', encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)  # write key pair objects as json formatted stream to json file
                f.write('\n')
            f.close
            if ((index+1) < files_len):
                print(f"\rProgress processing the JSON file: {round(((index+1)/files_len)*100, 2)}%", end="")
            else:
                print(f"\rProgress processing the JSON file: {round(((index+1)/files_len)*100, 2)}%")
        