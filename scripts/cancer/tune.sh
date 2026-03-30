export CUDA_VISIBLE_DEVICES=0,1,2,3
./scripts/cancer/tune/tune_ct.sh
./scripts/cancer/tune/tune_crn.sh
./scripts/cancer/tune/tune_gnet.sh
./scripts/cancer/tune/tune_rmsn.sh