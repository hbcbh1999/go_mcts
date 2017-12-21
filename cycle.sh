#/bin/bash

trap 'pkill -P $$' EXIT

#rm -r old_models
rm logs/*
rm -r data/cycle*
rm -r data/self*
rm -r saved_model


for ((x=0; x<200; x++))
{
	echo Cycle ${x} 
	for ((i=1; i<=4; i++)){
		mkdir -p data/self${i}
	}
	python play.py --cycle ${x} --black agent --white agent --ngames 250 --save --save_dir data/self1 >> logs/log_1 &
	python play.py --cycle ${x} --black agent --white agent --ngames 250 --save --save_dir data/self2 >> logs/log_2 &
	python play.py --cycle ${x} --black agent --white random --ngames 250 --save --save_dir data/self3 >> logs/log_3 &
	python play.py --cycle ${x} --black random --white agent --ngames 250 --save --save_dir data/self4 >> logs/log_4 &

	wait

	python train.py --new --epochs 10 --data_dir data/self1 data/self2 data/self3 data/self4 >> logs/log_train

	python play.py --black agent --white random --ngames 100 >> logs/log_bench_black &
	python play.py --black random --white agent --ngames 100 >> logs/log_bench_white &

	mkdir -p data/cycle${x}
	mv data/self* data/cycle${x}
}

