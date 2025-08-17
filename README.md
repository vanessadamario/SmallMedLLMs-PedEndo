# SmallMedLLs-PedEndo
Library for the creation of a Large Language Model based on Llama 3.1 8B model for consultations related to pediatric Type 1 Diabetes.

To use the code, run 
`python main.py --run pubmed --db pmc --startyear 2005 --endyear 2007`

--db can be either:
    1) 'pubmed' for PubMed
    2) 'pmc' for PubMed Central or 

Use whichever years you would like to search through.

Results can be found in the `data/` folder.