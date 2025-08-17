# SmallMedLLMs-PedEndo

Repository used for testing accuracy, consistency, self-assessment bias of small open-source large language models (LLMs) previously fine-tuned on clinical tasks. LLMs are evaluated on multiple choice questions from the Endocrine Self-Assessment Program using several metrics: 1) sensitivity to prompt, 2) sensitivity to letter token, 3) response variability in stochastic setting, 4) self-assessment bias. 

All the LLMs are open-source and publicly available on HuggingFace. In these experiments, models were tested as-is. Due to cluster configuration LLMs ran offline. We downloaded the models from HuggingFace, cloning the repository as 
`hf download FreedomIntelligence/HuatuoGPT-o1-8B`.
Below we report HuggingFace path and snapshot id.

[FreedomIntelligence/HuatuoGPT-o1-8B](https://huggingface.co/FreedomIntelligence/HuatuoGPT-o1-8B) `afc8b260e5b3dee9233863cf2de3080f3094442a`
[WaltonFuture/Diabetica-o1](https://huggingface.co/WaltonFuture/Diabetica-o1) `902b46e4354839ca37185f05c86029f1af5ecf27`
[WaltonFuture/Diabetica-7B](https://huggingface.co/WaltonFuture/Diabetica-7B) `7731a7a2153f564a5129b7fd795872eb59c50aa0`
[OpenMeditron/Meditron3-8B](https://huggingface.co/OpenMeditron/Meditron3-8B) `15914bcb040cd1a4f263afcd85b84f09ad2efd95`
[medicalai/ClinicalGPT-base-zh](https://huggingface.co/medicalai/ClinicalGPT-base-zh) `dbbea8f6ace2c98d1657825a0291c36cd17c5f78`
[medicalai/MedFound-7B](https://huggingface.co/medicalai/MedFound-7B) `7ed8b52f45d1ad504dbecae33577cc854c3d4278`
