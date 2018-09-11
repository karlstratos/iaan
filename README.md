# iaan

python run_pos.py --dynet-mem 512 --dynet-seed 42 --dynet-autobatch 1 /tmp/iaantemp data/pos/orig/en.words.1000 --gold data/pos/orig/en.tags.1000 --train --arch default --zsize 3 --wdim 1 --cdim 1 --jdim 0 --lrate 0.001 --drate 0.1 --epochs 10 --batch 80 --metric m2o --width 1 --loss lb

python run_pos.py --dynet-mem 512 --dynet-seed 42 --dynet-autobatch 1 /scratch/scratch/pos_tuned_m2o_en45_wdim200_cdim100_width2_epoch30_batch80_default_lb_lrate0.001_drate0.1 data/pos/orig/en.words --gold data/pos/orig/en.tags --train --verbose --arch default --zsize 45 --wdim 200 --cdim 100 --jdim 0 --lrate 0.001 --drate 0.1 --epochs 30 --batch 80 --metric m2o --width 2 --loss lb