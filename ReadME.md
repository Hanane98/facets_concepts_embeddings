# Facets_concept_embeddings
The code and datasets of "Modelling Commonsense Commonalities with Multi-Facet Concept Embeddings" presented in ACL 2024.

# Requirements
- Python 3.7
- transformers == 4.18.0
- scikit-learn == 1.0.2
- pytorch-metric-learning == 1.0.0
- collections == 3.3
- scipy == 1.7.3


# Usage
- **Step 1**: Start first by training the model to learn the facets by running the following command:
```
python ./src/facet_biencoder/train_facet_biencoder.py -cp_fn ./data/<dataset_name> train_cp_pairs.tsv -pf_fn ./data/<dataset_name>/train_pf_pairs.tsv -epoch 100 -lr 2e-5 -bsz 32 -model_dir ./trained_model -bert_version <bert_or_roberta_version> -weight_decay 1e-2 -max_seq_len 32 -model_prefix <bert_or_roberta_version>_facet_biencoder_<dataset_name> -tau 0.07 -m_per_class 2
```
where: `<bert_or_roberta_version>` is the name of the BERT or ROBERTA version chose, and `<dataset_name>` is the name of the chosen dataset (`chatgpt`, `conceptnet`, or `conceptnet+gpt`).

- **Step 2**: After training the model, get the embeddings. We show an example of how to do this on the McRae dataset:
```
python ./src/facet_biencoder/get_embeddings.py -in_fn data/McRae/abstract.txt -in_data_type concept -model_fn trained_model/<bert_or_roberta_version>_facet_biencoder_<dataset_name>_2e-05_20_0.07.pt -emb_fn embeddings/<name_of_the_obtained_embeddings>.pkl -bsz 32 -max_seq_len 32 -bert_version <bert_or_roberta_version>-tau 0.07
```
- **Step 3:** Facet clustering (choose the number of facets you'd like to test on and construct the facets spaces).
Do the same previous step to get embeddings of the properties of the dataset used for training by specifying the path of  -in_fn data/<dataset_name>/props.txt -in_data_type property, then run the command:
```
python ./src/facet_clustering.py  --input embeddings/<name_of_extracted_embeddings> --output facet_15_centroids/<dataset_name>_props_kmeans_centroids_<bert_or_roberta_version>.pt --method kmeans --clusters 15 
```

- **Step 4:** Get the masked embeddings of concepts (masking using facet embeddings obtained in Step 3):
```
python src/masked_con.py --centroids facet_15_centroids/<dataset_name>_props_kmeans_centroids_<bert_or_roeberta_version>.pt --in_embs embeddings/<name_of_the_obtained_embeddings>.pkl  --output embeddings/McRae/masked_con/15facets/<bert_or_roeberta_version>_facet
```
- **Step 5**: Testing 
i. Outlier detection using zscore or pscore
```
python src/outlier_detection_test.py <path_for_test_embeddings> data/McRae/3pos_7neg <path_to_save_csv_results_for_each_property> <path_to_save_csv_results_for_all_properties> <minus
_or_division>
```
You can add the `--zscore` option at the end of the command if you want to use zscore.
ii. Property prediction:
on the McRae dataset:
```
# concept split
python ./src/prop_prediction/facet_biencoder/finetune_cv.py -bert_version <bert_or_roberta_version> -model_fn trained_model/<bert_or_roberta_version>_facet_biencoder_<dataset_name>_2e-05_20_0.07.pt -epoch 100 -lr 2e-5 -bsz 32 -model_dir ./trained_models/ -max_seq_len 32 -model_prefix con_mcrae_finetuned_facet_biencoder -train_file_path ./data/extended_mcrae/train_mcrae.tsv -test_file_path ./data/extended_mcrae/test_mcrae.tsv -log_dir mcrae_con -load_pretrained -res_fn ./results/<dataset_name>_<bert_or_roberta_version>/mcrae_con.txt -tau 0.07 -test_mode con 

# property split
python ./src/prop_prediction/facet_biencoder/finetune_cv.py -bert_version <bert_or_roberta_version> -model_fn trained_model/<bert_or_roberta_version>_facet_biencoder_<dataset_name>_2e-05_20_0.07.pt -epoch 100 -lr 2e-5 -bsz 32 -model_dir ./trained_models/ -max_seq_len 32 -model_prefix pcv_mcrae_finetuned_facet_biencoder -train_file_path ./data/extended_mcrae/train_mcrae.tsv -test_file_path ./data/extended_mcrae/test_mcrae.tsv -log_dir mcrae_pcv -load_pretrained -res_fn ./results/<dataset_name>_<bert_or_roberta_version>/mcrae_pcv.txt -do_cv -cv_type model_evaluation_property_split -tau 0.07 -test_mode pcv

# concept property split
python ./src/prop_prediction/facet_biencoder/finetune_cv.py -bert_version <bert_or_roberta_version> -model_fn trained_model/<bert_or_roberta_version>_facet_biencoder_<dataset_name>_2e-05_20_0.07.pt -epoch 100 -lr 2e-5 -bsz 32 -model_dir ./trained_models/ -max_seq_len 32 -model_prefix cpcv_mcrae_finetuned_facet_biencoder -train_file_path ./data/extended_mcrae/train_mcrae.tsv -test_file_path ./data/extended_mcrae/test_mcrae.tsv -log_dir mcrae_cpcv -load_pretrained -res_fn ./results/<dataset_name>_<bert_or_roberta_version>/mcrae_cpcv.txt -do_cv -cv_type model_evaluation_concept_property_split -tau 0.07 -test_mode cpcv
```

on the CSLB dataset:
```
# concept split
python ./src/prop_prediction/facet_biencoder/finetune_cv.py -bert_version <bert_or_roberta_version> -model_fn trained_model/<bert_or_roberta_version>_facet_biencoder_<dataset_name>_2e-05_20_0.07.pt -epoch 100 -lr 2e-5 -bsz 32 -model_dir ./trained_models/ -max_seq_len 32 -model_prefix con_mcrae_finetuned_facet_biencoder -train_file_path ./data/CSLB/20_neg_cslb_train_pos_neg_data.tsv -test_file_path ./data/CSLB/20_neg_cslb_test_pos_neg_data.tsv -log_dir cslb_con -load_pretrained -res_fn ./results/<dataset_name>_<bert_or_roberta_version>/cslb_con.txt -tau 0.07 -test_mode con

# property split
python ./src/prop_prediction/facet_biencoder/finetune_cv.py -bert_version <bert_or_roberta_version> -model_fn trained_model/<bert_or_roberta_version>_facet_biencoder_<dataset_name>_2e-05_20_0.07.pt -epoch 100 -lr 2e-5 -bsz 32 -model_dir ./trained_models/ -max_seq_len 32 -model_prefix pcv_mcrae_finetuned_facet_biencoder -train_file_path ./data/CSLB/20_neg_cslb_train_pos_neg_data.tsv -test_file_path ./data/CSLB/20_neg_cslb_test_pos_neg_data.tsv -log_dir cslb_pcv -load_pretrained -res_fn ./results/<dataset_name>_<bert_or_roberta_version>/cslb_pcv.txt -do_cv -cv_type model_evaluation_property_split -tau 0.07 -test_mode pcv

# concept property split
python ./src/prop_prediction/facet_biencoder/finetune_cv.py -bert_version <bert_or_roberta_version> -model_fn trained_model/<bert_or_roberta_version>_facet_biencoder_<dataset_name>_2e-05_20_0.07.pt -epoch 100 -lr 2e-5 -bsz 32 -model_dir ./trained_models/ -max_seq_len 32 -model_prefix cpcv_mcrae_finetuned_facet_biencoder -train_file_path ./data/CSLB/20_neg_cslb_train_pos_neg_data.tsv -test_file_path ./data/CSLB/20_neg_cslb_test_pos_neg_data.tsv -log_dir cslb_cpcv -load_pretrained -res_fn ./results/<dataset_name>_<bert_or_roberta_version>/cslb_cpcv.txt -do_cv -cv_type model_evaluation_concept_property_split -tau 0.07 -test_mode cpcv
```
# Citation
```
@article{kteich2024modelling,
  title={Modelling Commonsense Commonalities with Multi-Facet Concept Embeddings},
  author={Kteich, Hanane and Li, Na and Chatterjee, Usashi and Bouraoui, Zied and Schockaert, Steven},
  journal={arXiv preprint arXiv:2403.16984},
  year={2024}
}
```
