import argparse
from datetime import datetime

parser = argparse.ArgumentParser()

parser.add_argument('--run', type=str, choices=['ncbi', 'inference'], required=True)
parser.add_argument('--local', action='store_true', help='It runs on GCP by default')

parser.add_argument('--db', type=str, default='pmc', choices=['pubmed', 'pmc'], required=False)
parser.add_argument('--startyear', type=int, required=False)
parser.add_argument('--endyear', type=int, default=datetime.today().year, required=False)
parser.add_argument('--output_base_dir', type=str, default="data/XML", required=False)
parser.add_argument('--output_base_extr', type=str, default="data/extracted_files", required=False)
parser.add_argument('--output_filename', type=str, default="output_cleaned_xmls.json", required=False)
parser.add_argument('--maxresults', type=int, default=10000, required=False)

parser.add_argument('--model', type=str, default='llama3.1', choices=['llama3.1', 'medfound7B', 'deepseekr1-1.5B', 'deepseekr1-7B',
                                                                      'deepseekr1-8B', 'deepseekr1-14B', 'huatuo-o1', 'diabetica-7B',
                                                                      'diabetica-o1', 'medical-llama3-8B', 'ufal', 'meditron3-8B',
                                                                      'biomed-multimodal', 'clinical-chatgpt', 'clinicalBERT',
                                                                      'meditron7B', 'biomed-contact-doc1B', 'biomed-contact-doc8B'], 
                                                                      required=False)
parser.add_argument('--temperature', type=float, default=0.1, required=False)
parser.add_argument('--max_tokens', type=int, default=800, required=False)
parser.add_argument('--top_p', type=float, default=0.9, required=False)
parser.add_argument('--top_k', type=int, default=75, required=False)
parser.add_argument('--repetition_penalty', type=float, default=1.2, required=False)
parser.add_argument('--length_penalty', type=float, default=1.0, required=False)
parser.add_argument('--do_sample', action='store_true', help='True if using sampling. Set to False by default')
parser.add_argument('--num_return_sequences', type=float, default=1, required=False)
parser.add_argument('--test_data', type=str, default='esap')
parser.add_argument('--prompt_template', type=str, default='001')
parser.add_argument('--eval_reasoning', action='store_true', help="Load model and evaluate reasoning capabilities")
parser.add_argument('--model_to_eval', type=str, default="huatuo-o1")
parser.add_argument('--esap_last', action='store_true', help="Load model and evaluate reasoning capabilities")
parser.add_argument('--experiment_id', type=int, required=False, help="Only when eval_reasoning is True, specify output filename")
parser.add_argument('--eliminate_letter_token', action='store_true', help='If True, A-E options are eliminated.')


FLAGS = parser.parse_args()


def create_PubMed_data():
    from preprocessing.query_pubmed import QueryNCBIManager, XMLConverter
    from preprocessing.query_pubmed import QUERY_STRING

    QueryDBS = QueryNCBIManager(QUERY_STRING,
                                FLAGS.startyear,
                                FLAGS.endyear,
                                database_system=FLAGS.db,
                                output_base_dir=FLAGS.output_base_dir,
                                max_results=FLAGS.maxresults)
    QueryDBS.search()
    files = QueryDBS.files

    XMLExtract = XMLConverter(files,
                              database_system=FLAGS.db,
                              output_base_dir=FLAGS.output_base_dir,
                              output_base_extr=FLAGS.output_base_extr,
                              output_json=FLAGS.output_filename)
    XMLExtract.save_metadata_cleaned_xmls()    


def run_llm_inference():
    from inference.run_llm_automodel import run_model
    run_model(local=FLAGS.local, model_label=FLAGS.model, temperature=FLAGS.temperature, max_tokens=FLAGS.max_tokens, 
              top_p=FLAGS.top_p, top_k=FLAGS.top_k, repetition_penalty=FLAGS.repetition_penalty, 
              length_penalty=FLAGS.length_penalty, do_sample=FLAGS.do_sample, num_return_sequences=FLAGS.num_return_sequences,
              test_data=FLAGS.test_data, prompt_id=FLAGS.prompt_template, eval_reasoning=FLAGS.eval_reasoning, 
              experiment_id=FLAGS.experiment_id, model_to_eval=FLAGS.model_to_eval, esap_last=FLAGS.esap_last,
              eliminate_letter_token=FLAGS.eliminate_letter_token)


switcher = {
    'ncbi': create_PubMed_data,
    'inference': run_llm_inference
    }

switcher[FLAGS.run]()