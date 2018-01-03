#/bin/bash

trap 'pkill -P $$' EXIT

#rm -r old_models
rm logs/*
rm -r data/cycle*
rm -r data/self*
rm -r saved_model


ngames=5
for ((x=0; x<200; x++))
{
	echo Cycle ${x} 
	for ((i=1; i<=4; i++)){
		mkdir -p data/self${i}
		python3 play.py --cycle ${x} --black agent --white agent --ngames ${ngames} --save --save_dir data/self${i}/ >> logs/log_${i} &
	}

	wait

	python3 train.py --new --epochs 10 --data_dir data/self1 data/self2 data/self3 data/self4 >> logs/log_train

	python3 play.py --black agent --white random --ngames 100 >> logs/log_bench_black &
	python3 play.py --black random --white agent --ngames 100 >> logs/log_bench_white &

	mkdir -p data/cycle${x}
	mv data/self* data/cycle${x}
}

