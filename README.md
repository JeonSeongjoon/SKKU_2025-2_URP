# SKKU_2025-2_URP
This is a github repository for SKKU 2025-2 URP.

The main theme of this research is knowledge distillation(KD) of reasoning ability from huge LLM to small models.
We created the dataset "KSAT_LEET_prob" by scraping the problems of KSAT and LEET that is freely accessible by internet.

The dataset is used for identifying the reasoning ability of small models and making the teacher labels from the GPT-model.
(The dataset is not provided because it is owned by Korean Instituion of Curriculum & Evaluation(한국교육과정평가원) and The Association of Korean Law Schools.)

We share the result of the research below

<img width="480" height="360" alt="train_history_fig_log_kakaocorp-kanana-1 5-8b-instruct-2505_epochs_5_lr_0 0001_(trained)" src="https://github.com/user-attachments/assets/ba66f000-694d-4773-940c-ba31fbac25b8" /><br>
<img width="480" height="360" alt="train_history_fig_log_LGAI-EXAONE-EXAONE-3 5-7 8B-Instruct_epochs_5_lr_0 0001_(trained) (1)" src="https://github.com/user-attachments/assets/a90757bd-004f-4a48-860e-f756e172f52b" /><br>
<img width="480" height="360" alt="train_history_fig_log_skt-A X-4 0-Light_epochs_5_lr_0 0001_(trained)" src="https://github.com/user-attachments/assets/bfedca06-cd3e-4111-9e19-b1636916ee1b" /><br>
[Fig1 : epoch_loss_graph]


The whole process is worked on google colab environment.
