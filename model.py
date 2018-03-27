#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Store XGBoost Model Parameters
"""

import xgboost as xgb


def get_models(name):
    if name == 'basic':
        a_clf = xgb.XGBClassifier(
            max_depth=3,
            learning_rate=.1,
            n_estimators=10,
            silent=True,
            objective="binary:logistic",
            nthread=-1,
            gamma=0,
            min_child_weight=1,
            max_delta_step=0,
            subsample=1,
            colsample_bytree=1,
            colsample_bylevel=1,
            reg_alpha=0,
            reg_lambda=1,
            scale_pos_weight=1,
            base_score=.5,
            seed=0
        )
        v_clf = xgb.XGBClassifier(
            max_depth=3,
            learning_rate=.1,
            n_estimators=10,
            silent=True,
            objective="binary:logistic",
            nthread=-1,
            gamma=0,
            min_child_weight=1,
            max_delta_step=0,
            subsample=1,
            colsample_bytree=1,
            colsample_bylevel=1,
            reg_alpha=0,
            reg_lambda=1,
            scale_pos_weight=1,
            base_score=.5,
            seed=0
        )
    elif name == 'eeg':
        a_clf = xgb.XGBClassifier(
            max_depth=4,
            learning_rate=.17,
            n_estimators=81,
            silent=True,
            objective="binary:logistic",
            nthread=-1,
            gamma=0,
            min_child_weight=.81,
            max_delta_step=0,
            subsample=1,
            colsample_bytree=.88,
            colsample_bylevel=1,
            reg_alpha=0,
            reg_lambda=1,
            scale_pos_weight=1,
            base_score=.5,
            seed=0
        )
        v_clf = xgb.XGBClassifier(
            max_depth=5,
            learning_rate=.1,
            n_estimators=28,
            silent=True,
            objective="binary:logistic",
            nthread=-1,
            gamma=0,
            min_child_weight=1,
            max_delta_step=.87,
            subsample=1,
            colsample_bytree=1,
            colsample_bylevel=1,
            reg_alpha=0,
            reg_lambda=1,
            scale_pos_weight=1,
            base_score=.5,
            seed=0
        )
    elif name == 'eeg_new':
        a_clf = xgb.XGBClassifier(
            max_depth=5,
            learning_rate=.1,
            n_estimators=60,
            silent=True,
            objective="binary:logistic",
            nthread=-1,
            gamma=0,
            min_child_weight=1,
            max_delta_step=0,
            subsample=1,
            colsample_bytree=1,
            colsample_bylevel=1,
            reg_alpha=0,
            reg_lambda=1,
            scale_pos_weight=.99,
            base_score=.38,
            seed=0
        )
        v_clf = xgb.XGBClassifier(
            max_depth=2,
            learning_rate=.1,
            n_estimators=80,
            silent=True,
            objective="binary:logistic",
            nthread=-1,
            gamma=0,
            min_child_weight=.99,
            max_delta_step=0,
            subsample=1,
            colsample_bytree=.29,
            colsample_bylevel=1,
            reg_alpha=0,
            reg_lambda=1,
            scale_pos_weight=.97,
            base_score=.54,
            seed=0
        )
    elif name == 'ecg':
        a_clf = xgb.XGBClassifier(
            max_depth=1,
            learning_rate=.1,
            n_estimators=3,
            silent=True,
            objective="binary:logistic",
            nthread=-1,
            gamma=0,
            min_child_weight=.8,
            max_delta_step=.45,
            subsample=.75,
            colsample_bytree=.62,
            colsample_bylevel=1,
            reg_alpha=.1,
            reg_lambda=.85,
            scale_pos_weight=1,
            base_score=.5,
            seed=param
        )
        v_clf = xgb.XGBClassifier(
            max_depth=2,
            learning_rate=.1,
            n_estimators=61,
            silent=True,
            objective="binary:logistic",
            nthread=-1,
            gamma=0,
            min_child_weight=1,
            max_delta_step=1.5,
            subsample=1,
            colsample_bytree=1,
            colsample_bylevel=1,
            reg_alpha=0,
            reg_lambda=1,
            scale_pos_weight=1,
            base_score=.5,
            seed=0
        )
    elif name == 'ecg_new':
        a_clf = xgb.XGBClassifier(
            max_depth=4,
            learning_rate=.1,
            n_estimators=99,
            silent=True,
            objective="binary:logistic",
            nthread=-1,
            gamma=3.5,
            min_child_weight=1.18,
            max_delta_step=0,
            subsample=1,
            colsample_bytree=1,
            colsample_bylevel=1,
            reg_alpha=.65,
            reg_lambda=1,
            scale_pos_weight=1,
            base_score=.5,
            seed=0
        )
        v_clf = xgb.XGBClassifier(
            max_depth=4,
            learning_rate=.1,
            n_estimators=47,
            silent=True,
            objective="binary:logistic",
            nthread=-1,
            gamma=0,
            min_child_weight=1,
            max_delta_step=0,
            subsample=1,
            colsample_bytree=1,
            colsample_bylevel=.7,
            reg_alpha=0,
            reg_lambda=1,
            scale_pos_weight=1,
            base_score=.48,
            seed=0
        )
    elif name == 'gsr':
    elif name == 'gsr_new':
    elif name == 'all':
    elif name == 'all_new':

    return a_clf, v_clf


def main():
    """ Main function """


if __name__ == '__main__':

    main()
