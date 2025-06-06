
dir="GSH"
singularity exec \
     --nv \
     --bind $PWD/:/root/af_input \
     --bind $PWD/:/root/af_output \
     --bind /home/nsingh/Desktop/alphafold_files:/root/models \
     --bind /ponderosa/tank/alphafolddatabases/:/root/public_databases \
     /home/nsingh/Desktop/alphafold_files/alphafold3.sif \
     python run_alphafold.py \
     --json_path=/root/af_input/fold_input.json \
     --model_dir=/root/models \
     --db_dir=/root/public_databases \
     --output_dir=/root/af_output