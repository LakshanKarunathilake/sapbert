CUDA_VISIBLE_DEVICES=$1 python3 train.py \
	--model_dir "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext" \
	--train_dir "/mnt/exssd2/share/lakshan/sapbert/training_data/training_file_umls2025aa_en_uncased_no_dup_pairwise_pair_th50.txt" \
	--output_dir output/sapbert-modernbert-adapter \
	--use_adapters \
    --adapter_config "lora" \
	--wandb_project "sapbert-adapter" \
	--wandb_run_name "umls-modern-bert-adapter" \
	--use_cuda \
	--epoch 2 \
	--train_batch_size 256 \
	--learning_rate 2e-5 \
	--max_length 25 \
	--checkpoint_step 999999 \
	--parallel \
	--amp \
	--pairwise \
	--random_seed 33 \
	--loss ms_loss \
	--use_miner \
	--type_of_triplets "all" \
	--miner_margin 0.2 \
	--agg_mode "cls"

# CUDA_VISIBLE_DEVICES=$1 python3 train.py \
# 	--model_dir "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext" \
# 	--train_dir "/mnt/exssd1/share/lakshan/sapbert/training_data/wikidata/output/aliases_positive_pairs.txt" \
# 	--output_dir output/sapbert-full \
# 	--wandb_project "sapbert-wikidata" \
# 	--wandb_run_name "full-model" \
# 	--use_cuda \
# 	--epoch 1 \
# 	--train_batch_size 256 \
# 	--learning_rate 2e-5 \
# 	--max_length 25 \
# 	--checkpoint_step 999999 \
# 	--parallel \
# 	--amp \
# 	--pairwise \
# 	--random_seed 33 \
# 	--loss ms_loss \
# 	--use_miner \
# 	--type_of_triplets "all" \
# 	--miner_margin 0.2 \
# 	--agg_mode "cls"


	# CUDA_VISIBLE_DEVICES=$1 python3 train.py \
	# --model_dir "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext" \
	# --train_dir "/mnt/exssd1/share/lakshan/sapbert/training_data/training_file_umls2025aa_en_uncased_no_dup_pairwise_pair_th50.txt" \
	# --output_dir output/sapbert-mesh-full \
	# --wandb_project "sapbert-mesh-full" \
	# --wandb_run_name "full-model" \
	# --use_cuda \
	# --epoch 1 \
	# --train_batch_size 256 \
	# --learning_rate 2e-5 \
	# --max_length 25 \
	# --checkpoint_step 999999 \
	# --parallel \
	# --amp \
	# --pairwise \
	# --random_seed 33 \
	# --loss ms_loss \
	# --use_miner \
	# --type_of_triplets "all" \
	# --miner_margin 0.2 \
	# --agg_mode "cls"

	# CUDA_VISIBLE_DEVICES=$1 python3 train.py \
	# --model_dir "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext" \
	# --train_dir "/mnt/exssd1/share/lakshan/sapbert/training_data/training_file_umls2025aa_en_uncased_no_dup_pairwise_pair_th50.txt" \
	# --output_dir output/sapbert-mesh-adapter-5-epoch \
	# --use_adapters \
    # --adapter_config "pfeiffer" \
	# --wandb_project "sapbert-adapter" \
	# --wandb_run_name "umls-5-epoch" \
	# --use_cuda \
	# --epoch 5 \
	# --train_batch_size 256 \
	# --learning_rate 2e-5 \
	# --max_length 25 \
	# --checkpoint_step 999999 \
	# --parallel \
	# --amp \
	# --pairwise \
	# --random_seed 33 \
	# --loss ms_loss \
	# --use_miner \
	# --type_of_triplets "all" \
	# --miner_margin 0.2 \
	# --agg_mode "cls"


# with type weights
# CUDA_VISIBLE_DEVICES=$1 python3 train.py \
# 	--model_dir "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext" \
# 	--train_dir "/mnt/exssd2/share/lakshan/sapbert/training_data/training_file_umls2025aa_en_uncased_no_dup_pairwise_pair_th50_tui.txt" \
# 	--output_dir output/typed-training/sapbert-umls-adapter \
# 	--use_adapters \
#     --adapter_config "pfeiffer" \
# 	--wandb_project "sapbert-adapter" \
# 	--wandb_run_name "typed-pfeiffer" \
# 	--use_cuda \
# 	--epoch 1 \
# 	--train_batch_size 256 \
# 	--learning_rate 2e-5 \
# 	--max_length 25 \
# 	--checkpoint_step 999999 \
# 	--parallel \
# 	--amp \
# 	--pairwise \
# 	--random_seed 33 \
# 	--loss ms_loss \
# 	--use_miner \
# 	--type_of_triplets "all" \
# 	--miner_margin 0.2 \
# 	--enable_types \
# 	--agg_mode "cls" \
# 	--use_type_weights --w_type 0.5 --tau_type 0.2 \
#   	--use_sty_aux --lambda_sty 0.1 \
#   	--tui2idx_json /mnt/exssd2/share/lakshan/sapbert/training_data/tui2idx.json