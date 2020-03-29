#!/usr/bin/env bash
python ./mlpipeline/predict.py \
--is_train False \
--model_name lgb \
--pred_start_date 2018-09-01 \
--pred_end_date 2018-09-30 \
--use_standard False \
--scaler '' \
--use_log False \
--num_processes 8 \
