import os
import re
import sys
import json
import unicodedata
from bs4 import BeautifulSoup






# # ====================================================================
# # Rather than make changes to your function, I opted to copy, paste, and rename it below
# # because I didn't know what changes it would need. I kept this function as a reference.
# def extract_sections_with_loose_intro(xml_file):
#     # Read the XML file
#     with open(xml_file, "r", encoding="utf-8") as file:
#         soup = BeautifulSoup(file, "xml")
    
#     extracted_content = {}

#     body = soup.find("body")  # Find the <body> tag
#     if not body:
#         return extracted_content  # No body found, return empty
    
#     loose_text = [] # Check for direct text or content immediately under <body>
#     loose_text_flag = False

#     for child in body.contents:
#         if child.name == "sec":
#             break  
#         else: # For those papers that do not have an introduction section
#             if str(child) != '\n':
#                 loose_text_flag = True
#                 loose_text.append(preprocess_text(str(child)))
#     loose_text = " ".join(loose_text)

#     if loose_text_flag:
#         clean_introduction  = re.sub(r"<[^>]+>", "", loose_text) # This removes formatting as in xml files
#         extracted_content["introduction"] = clean_introduction.strip()
        
#     for sec in body.find_all("sec"):  # Process <sec> elements 
#         title = sec.find("title")
#         title_text = title.get_text(strip=True).lower() if title else None
#         paragraphs = [preprocess_text(p.text.strip()) for p in sec.find_all("p")]
#         extracted_content[title_text] = " ".join(paragraphs)

#     return extracted_content


# Remove the reference numbers and surrounding [] and () from the text
# Cleans <xref> tags, surrounding brackets [] and (), and separators like dashes or commas from the XML content.
# The <xref> elements contain the reference numbers within the XML file.
def clean_xref_tags(xml_content):
    xml_content = re.sub(  #remove patterns like [ <xref>1</xref> – <xref>3</xref> ] and ( <xref>1</xref> – <xref>3</xref> )
        r"\s*[\[\(]?\s*<xref[^>]*>.*?</xref>\s*(?:[-–,]\s*<xref[^>]*>.*?</xref>)*\s*[\]\)]?",
        "",
        xml_content,
    )
    xml_content = re.sub(r"<xref[^>]*>.*?</xref>", "", xml_content)  # remove standalone <xref> tags if any remain
    return xml_content


def extract_sections_within_body(xml_file, db):

    from preprocessing.query_pubmed import extract_pmc_article_metadata
    from preprocessing.query_pubmed import extract_pubmed_article_metadata

    with open(xml_file, "r", encoding="utf-8") as f:  # read the XML file content
        xml_content = f.read()

    # clean the XML content by removing unwanted <xref> tags and structures before converting the XML using BeautifulSoup
    cleaned_xml_content = clean_xref_tags(xml_content)

    soup = BeautifulSoup(cleaned_xml_content, "xml")  # parse the cleaned XML with BeautifulSoup

    # We need to check if any <xref> remain calling the clean_xref_tags function above
    #for xref in soup.find_all('xref'):
    #    xref.decompose()

    # Remove the tables which are depicted by the <table-wrap> elements
    for table_wrap in soup.find_all('table-wrap'):
        table_wrap.decompose()

    # Remove the figures which are depicted by the <fig> elements
    for fig in soup.find_all('fig'):
        fig.decompose()

    # Extract article metadata
    if (db == 'pmc'):
        metadata = extract_pmc_article_metadata(soup)
    else:
        metadata = extract_pubmed_article_metadata(soup)


    # Initialize the dictionary to store the output
    abstract_content = {}

    # Process the <Abstract> element
    abstract = soup.find("Abstract") or soup.find("abstract") or soup.find("AbstractText")
    if abstract:

        #f.write(f"  - Section ID: {section.get('sec_id', 'N/A')}; Title: {section.get('title', 'N/A')};\n")
        #f.write(f"  - Label: {section.get('label', 'N/A')}; NLM Category: {section.get('nlm_category', 'N/A')};\n")

         # Initialize a default title for the first section
        previous_section_title = "Introduction"  # Default for the first section if no title exists
        
        # Dictionary to store section content

        # Process <sec> and <p> tags
        for element in abstract.find_all(['sec', 'p', 'AbstractText'], recursive=False):
            if element.name == 'sec':
                title_element = element.find("title")
                # Use the previous section title if no title exists, otherwise use the current title
                section_title = title_element.text.strip() if title_element and title_element.text.strip() else previous_section_title
                
                if title_element:
                    title_element.decompose()  # Remove the title tag to avoid duplication
                
                section_content = element.get_text(separator=" ", strip=True)
                abstract_content[section_title] = (abstract_content.get(section_title, "") + " " + preprocess_text(section_content)).strip()
                previous_section_title = section_title  # Update the previous section title
            elif element.name == 'p':
                section_title = previous_section_title  # Inherit the previous section title
                section_content = element.get_text(separator=" ", strip=True)
                abstract_content[section_title] = (abstract_content.get(section_title, "") + " " + preprocess_text(section_content)).strip()
            elif element.name == 'AbstractText':
                # Extract text directly from AbstractText element
                section_title = previous_section_title # Inherit the previous section title
                section_content = element.get_text(separator=" ", strip=True)
                abstract_content[section_title] = (abstract_content.get(section_title, "") + " " + preprocess_text(section_content)).strip()

    # Locate the <body> element
    body = soup.find("body")

    # Initialize the content dictionary
    body_content = {}
    if body:
        # Initialize a default title for the first section
        previous_section_title = "Introduction"  # Default for the first section if no title exists
        
        # Dictionary to store section content
        body_content = {}

        # Process <sec> and <p> tags
        for element in body.find_all(['sec', 'p'], recursive=False):
            if element.name == 'sec':
                title_element = element.find("title")
                # Use the previous section title if no title exists, otherwise use the current title
                section_title = title_element.text.strip() if title_element and title_element.text.strip() else previous_section_title
                
                if title_element:
                    title_element.decompose()  # Remove the title tag to avoid duplication
                
                section_content = element.get_text(separator=" ", strip=True)
                body_content[section_title] = (body_content.get(section_title, "") + " " + preprocess_text(section_content)).strip()
                previous_section_title = section_title  # Update the previous section title
            elif element.name == 'p':
                section_title = previous_section_title  # Inherit the previous section title
                section_content = element.get_text(separator=" ", strip=True)
                body_content[section_title] = (body_content.get(section_title, "") + " " + preprocess_text(section_content)).strip()

    # The key for the extracted content dictionary depends on the database.
    # PMC uses the 'pmc' key, while PubMed uses the 'pmid' key.
    key = None
    if db == 'pmc':
        key = metadata['pmc']
    else:
        key = metadata['pmid']
    
    extracted_content = {}
    # Build the output dictionary
    extracted_content[key] = {
        "pmid": metadata['pmid'],
        "pmc": metadata['pmc'],
        "doi": metadata['doi'],
        "article-title": metadata['article-title'],
        "year": metadata['year'],
        "abstract-content": abstract_content,
        "body-content": body_content
    }
    
    return extracted_content




def save_dataset_to_json(path_to_files, path_to_save, filenames, db, output_json='test.json'):

    files_length = 0
    if (filenames is not None):
        files_length = len(filenames)
    else:
        return
    
    if files_length == 0:
        return

    with open(os.path.join(path_to_save, output_json), 'w', encoding="utf-8") as f:  # Open the file one time. Use mode 'w' so that the file is recreated each time. Using 'a' will append to the previous file causing duplicate output
        for index, file in enumerate(filenames): # Loop through the filenames
            content_entry = extract_sections_within_body(os.path.join(path_to_files, file), db)
            json.dump(content_entry, f, indent=4, ensure_ascii=False)    #write key pair objects as json formatted stream to json file
            f.write('\n')
            
            if ((index+1) < files_length):
                print(f"\rProgress processing the JSON file: {round(((index+1)/files_length)*100, 2)}%", end="")
            else:
                print(f"\rProgress processing the JSON file: {round(((index+1)/files_length)*100, 2)}%")

        f.close
