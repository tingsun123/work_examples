#!/usr/bin/env python3
# pyre-strict
# Ting Sun - Sep. 2022, adapted code from orignal author: Dai Lu
## output add positive or negative direction.

import time

import numpy as np
import pandas as pd
#from src.attributionModel import ModelAttribution
from pandas import DataFrame
from typing import Tuple,Dict,List


class AttrShiftRCA(object):
    def __init__(self, dfs_dict, target, metric_dict, ModelClass,shapSampleIdx, n_iter=2,ShapSampleSize=10):
        """
        @dfs_dict: a dictionary of dataframes
        @target: target variable. For efficiency project: 'training_efficiency'
        @metric_dict: a dictionary of debugging metrics
        @ModelClass: ModelAttribution -- the tpot model trained with shap value calculated.
        @n_iter: number of iterations for the Separate Modeling
        @shapSampleIdx: sample index for control and regression period. (the index for the data frame which is also the root_workflow_run_id.)
        @ShapSampleSize: sample size for joint modeling, which is the sample size

        """
        #super().__init__(self,*args)
        self.dfs_dict = dfs_dict
        self.target = target
        self.metric_dict = metric_dict
        self.Model = ModelClass
        self.n_iter = n_iter
        self.shapSampleIdx=shapSampleIdx
        self.ShapSampleSize=ShapSampleSize

    def JointModel(self, metric_key:str) -> Tuple[DataFrame,Dict,DataFrame,DataFrame]:
        """
        Joint Modeling Attribution-Shift RCA.
        Fit one joint model for both control period and regression period together
        Obtain feature attribution from one joint models
        But aggregate observations in control period and regression period respectively to obtain different feature attribution
        Return a dataframe including feature attribution shift
        """

        attr_values, y, y_pred, fitted_mean,df_shap_joint = self.Model.get_shap_val(
            df=self.dfs_dict[metric_key + "_all"],
            target_Y=self.target,
            metrics=self.metric_dict[metric_key],
            shapSampleIdx=self.shapSampleIdx
        )


        mean_dic={}
        mean_dic['fitted_mean'] = fitted_mean
        # ctl_idx = np.where(
        #     np.in1d(
        #         #self.dfs_dict[metric_key + "_all"][self.shapSampleIdx,:].index,
        #         self.shapSampleIdx,
        #         self.dfs_dict[metric_key + "_ctl"].index
        #     )
        # )[0]
        # ctl_idx =  np.intersect1d(self.shapSampleIdx,
        #          self.dfs_dict[metric_key + "_ctl"].index)

        ctl_idx = list(set(self.shapSampleIdx).intersection(set(self.dfs_dict[metric_key + "_ctl"].index)))

        # reg_idx = np.where(
        #     np.in1d(
        #         #self.dfs_dict[metric_key + "_all"][self.shapSampleIdx,:].index,
        #         self.shapSampleIdx,
        #         self.dfs_dict[metric_key + "_reg"].index,
        #     )
        # )[0]
        reg_idx = list(set(self.shapSampleIdx).intersection(set(self.dfs_dict[metric_key + "_reg"].index)))

        #np.intersect1d(self.shapSampleIdx,
        #         self.dfs_dict[metric_key + "_reg"].index)

        # assert len(ctl_idx)>0
        # assert len(reg_idx) >0

        metricMean_ctl=df_shap_joint.loc[ctl_idx].mean();metricMean_reg=df_shap_joint.loc[reg_idx].mean()

        mean_dic['metricMean_ctl']=pd.DataFrame(metricMean_ctl,columns=['ctl_mean'])
        mean_dic['metricMean_reg']=pd.DataFrame(metricMean_reg,columns=['reg_mean'])

        #print('metricMean is :',metricMean_ctl,metricMean_reg)
        #print("predicted difference between two=", np.mean(y_pred[:self.ShapSampleSize]), np.mean(y_pred[self.ShapSampleSize:]))
        #print("predicted difference between two=", np.mean(y_pred[:self.ShapSampleSize]), np.mean(y_pred[self.ShapSampleSize:]))
        avg_ctl=np.mean(y.loc[ctl_idx])
        avg_reg=np.mean(y.loc[reg_idx])

        ctl_reg_change=(avg_reg-avg_ctl)/avg_ctl*100
        if ctl_reg_change>0:
            print(f"Average training efficiency increased from {avg_ctl:.2f} to {avg_reg:.2f}, around {ctl_reg_change:.1f}%" )
        else:
            print(f"Average training efficiency dropped from {avg_ctl:.2f} to {avg_reg:.2f}, around {ctl_reg_change:.1f}%" )

        mean_dic['pred_diff'] = np.mean(y_pred[self.ShapSampleSize:])-np.mean(y_pred[:self.ShapSampleSize])
        mean_dic['actual_diff'] = np.mean(y.loc[reg_idx])- np.mean(y.loc[ctl_idx])



        ctl_contrib = self.Model.get_contrib_val(
            shap_values=attr_values.loc[ctl_idx, :] , y_pred=y_pred[:self.ShapSampleSize]
        )
        ctl_contrib_direction=self.Model.get_contrib_direction(shap_values=attr_values.loc[ctl_idx, :], y_pred=y_pred[:self.ShapSampleSize])

        reg_contrib = self.Model.get_contrib_val(
            shap_values=attr_values.loc[reg_idx, :], y_pred=y_pred[self.ShapSampleSize:])

        reg_contrib_direction= self.Model.get_contrib_direction(shap_values=attr_values.loc[ctl_idx, :], y_pred=y_pred[self.ShapSampleSize:])




        contrib = pd.DataFrame(
            {
                "Candidates": self.metric_dict[metric_key],
                "Level": [metric_key] * len(self.metric_dict[metric_key]),
                "ctl": ctl_contrib,
                "ctl_direction":ctl_contrib_direction,
                "reg": reg_contrib,
                "reg_direction": reg_contrib_direction
            }
        )

        contrib["relative_change%"]=(contrib["reg"] - contrib["ctl"])/ contrib["ctl"] * 100
        contrib["absolute_change"]=(contrib["reg"] - contrib["ctl"])


        return contrib.sort_values("absolute_change", ascending=False),mean_dic,attr_values.loc[ctl_idx],attr_values.loc[reg_idx]


    # def SeparateModel(self, metric_key:str) -> Tuple[DataFrame,Dict]:
    #     """
    #     Separate Modeling Attribution-Shift RCA.
    #     Fit two separate models for control period and regression period
    #     Obtain feature attribution from two separate models
    #     Iterate for n_iter times to reduce model variation
    #     Return a dataframe including feature attribution shift
    #     """
    #     contrib_ctls, contrib_regs = [], []
    #     for _ in range(self.n_iter):
    #         attr_ctl, y_ctl,ypred_ctl,_ = self.Model.get_shap_val(
    #             df=self.dfs_dict[metric_key + "_ctl"],
    #             target_Y=self.target,
    #             metrics=self.metric_dict[metric_key],
    #             shapSampleIdx=self.shapSampleIdx[:self.ShapSampleSize]

    #         )
    #         contrib_ctl = pd.DataFrame(
    #             {
    #                 "Candidates": self.metric_dict[metric_key],
    #                 "Level": [metric_key] * len(self.metric_dict[metric_key]),
    #                 "ctl": self.Model.get_contrib_val(
    #                     shap_values=attr_ctl, y_pred=ypred_ctl
    #                 ),
    #                 "direction_ctl":self.Model.get_contrib_direction(
    #                     shap_values=attr_ctl, y_pred=ypred_ctl
    #                 ),
    #             }
    #         )
    #         contrib_ctls.append(contrib_ctl)

    #         attr_reg, y_reg,ypred_reg , _= self.Model.get_shap_val(
    #             df=self.dfs_dict[metric_key + "_reg"],
    #             target_Y=self.target,
    #             metrics=self.metric_dict[metric_key],
    #             shapSampleIdx=self.shapSampleIdx[self.ShapSampleSize:]
    #         )
    #         contrib_reg = pd.DataFrame(
    #             {
    #                 "Candidates": self.metric_dict[metric_key],
    #                 "Level": [metric_key] * len(self.metric_dict[metric_key]),
    #                 "reg": self.Model.get_contrib_val(
    #                     shap_values=attr_reg, y_pred=ypred_reg
    #                 ),
    #                 "direction_reg":self.Model.get_contrib_direction(
    #                     shap_values=attr_ctl, y_pred=ypred_reg
    #                 )
    #             }
    #         )
    #         contrib_regs.append(contrib_reg)

    #     print("==================== mean of the joint data from separate model ==", np.mean(list(y_ctl)+list(y_reg)))
    #     print("==================== mean of control period  ==", np.mean(list(y_ctl)))
    #     print("==================== mean of regression period  ==", np.mean(list(y_reg)))

    #     # Get mean attribution values of ensembles
    #     contrib_ctl = (
    #         pd.concat(contrib_ctls)
    #         .groupby(["Candidates", "Level"])[["ctl"]]
    #         .mean()
    #         #.agg({'healthy': ['mean'], 'direction': 'sum'})
    #         .reset_index()
    #     )
    #     contrib_reg = (
    #         pd.concat(contrib_regs)
    #         .groupby(["Candidates", "Level"])[["reg"]]
    #         .mean()
    #         #.agg({'healthy': ['mean'], 'direction': 'sum'})
    #         .reset_index()
    #     )

    #     contrib = contrib_ctl.merge(contrib_reg, on=["Level", "Candidates"])

    #     contrib["absolute_change"] = (
    #         (contrib["reg"] - contrib["ctl"])

    #     )
    #     contrib["relative_change%"] = (
    #         (
    #             (contrib["reg"] - contrib["ctl"])
    #             / (contrib["ctl"] + 1e-9)
    #         )
    #         * 100
    #         )
    #     mean_dic={}
    #     mean_dic['fitted_mean'] = np.mean(list(y_ctl)+list(y_reg))

    #     mean_dic['pred_diff'] = np.mean(ypred_reg) -np.mean(ypred_ctl)
    #     mean_dic['actual_diff'] = np.mean(y_reg) - np.mean(y_ctl)


    #     return contrib.sort_values("absolute_change", ascending=False),mean_dic,attr_ctl,attr_reg



    def RCA(self, method:str, pc_threshold:float) -> Tuple[List,DataFrame]:
        """
        User specify which modeling method to use for RCA: 'Joint', 'Separate', 'Both'
        The default percentage threshold for significant feature attribution shift is 0.1 for absolute change.
        Return a list of attibution-shift dataframes, and print out the results

        @ the threshold to print the contribution output
        """
        contrib_1, contrib_2 = [], []
        contrib_df1, contrib_df2 = pd.DataFrame(), pd.DataFrame()
        if method == "Joint" :
            for metric_key in self.metric_dict.keys():
                # print(
                #     f"==================== Joint-Modeling RCA for {metric_key} ===================="
                # )
                st = time.time()
                contrib, mean_dic,attr_ctl,attr_reg = self.JointModel(metric_key)
                contrib_1.append(contrib)
                run_time = str(np.round(time.time() - st, 2))
                # print(
                #     f"Finished in {run_time} seconds: Joint Modeling for {metric_key}"
                # )
            contrib_df1 = pd.concat(contrib_1).sort_values(
                "absolute_change" #, ascending=False
            )

            contri_joint = contrib_df1[["Candidates", "absolute_change"]]
            contri1 = contri_joint[
                np.abs(contri_joint["absolute_change"]) >= pc_threshold
            ].reset_index(drop=True)


            contri=(mean_dic['metricMean_ctl'].merge(mean_dic['metricMean_reg'],how='inner',left_index=True,right_index=True)).\
                    merge(contri1,how='inner',left_index=True, right_on='Candidates').sort_values(
                "absolute_change"
            )

            #### shap value calibration based on predicted_diff vs actual_diff
            # contri=MetricMean_df.merge(contri1,how='inner',left_index=True, right_on='Candidates').sort_values(
            #     "calibrated_change" #, ascending=False
            # )

            if mean_dic['actual_diff']*mean_dic['pred_diff']<0:
                raise ValueError('difference between actual and predicted are in different direction. Re-run model!')

            contri=contri.assign(calibrated_change = contri["absolute_change"] * mean_dic['actual_diff']/mean_dic['pred_diff'],\
                    metric_change_pct=(contri['reg_mean']-contri['ctl_mean'])/contri['ctl_mean'] *100)

            contri=contri.assign(contribution_pct=contri['calibrated_change']/mean_dic['actual_diff'] *100)\
            .assign(direction=lambda x: 'regression' if mean_dic['actual_diff']<0 else 'progression')
            contri=contri.iloc[contri['contribution_pct'].abs().argsort()[::-1]]


            printout_cols=['Candidates','ctl_mean','reg_mean','metric_change_pct','absolute_change','calibrated_change','contribution_pct','direction']



            joint_rc = [
                x[0] + " (efficiency)" for x in contri[["Candidates"]].values
            ]
            rcs = joint_rc

        # if method == "Separate" or method == "Both":
        #     for metric_key in self.metric_dict.keys():
        #         print(
        #             f"==================== Separate-Modeling RCA for {metric_key} ===================="
        #         )
        #         st = time.time()
        #         contrib,mean_dic = self.SeparateModel(metric_key)
        #         contrib_2.append(contrib)
        #         run_time = str(np.round(time.time() - st, 2))
        #         print(
        #             f"Finished in {run_time} seconds: Separate Modeling for {metric_key}"
        #         )
        #     contrib_df2 = pd.concat(contrib_2).sort_values(
        #         "absolute_change" #, ascending=False
        #     )

        #     contri_sep = contrib_df2[["Candidates", "Level", "absolute_change"]]
        #     contri = contri_sep[
        #         np.abs(contri_sep["absolute_change"]) >= pc_threshold
        #     ].reset_index(drop=True)

        #     #### shap value calibration based on predicted_diff vs actual_diff
        #     contri['calibrated_change'] = contri["absolute_change"] * mean_dic['actual_diff']/mean_dic['pred_diff']



        #     sep_rc = [
        #         x[0] + " (" + x[1] + ")" for x in contri[["Candidates", "Level"]].values
        #     ]
        #     rcs = sep_rc

        # if method == "Both":
        #     contri = contri_joint.merge(
        #         contri_sep, on=["Candidates", "Level"], how="outer"
        #     )
        #     contri.columns = [
        #         "Candidates",
        #         "Level",
        #         "relative_change% (joint)",
        #         "relative_change% (sep)",
        #     ]
        #     contri = (
        #         contri[
        #             (contri["relative_change% (joint)"] >= pc_threshold)
        #             | (contri["relative_change% (sep)"] >= pc_threshold)
        #         ]
        #         .sort_values("relative_change% (joint)", ascending=False)
        #         .reset_index(drop=True)
        #     )
        #     pri = set(sep_rc).intersection(joint_rc)
        #     rcs = list(pri) + list(set(joint_rc) - pri) + list(set(sep_rc) - pri)

        # if len(rcs) > 0:
        #     print(
        #         f"\n Our analysis suggests investigating the following root causes: {rcs}.\n"
        #     )
        #     print(contri.round(2))
        # else:
        #     print("We did't find any root causes among the specified metrics.")

        return rcs, contri[printout_cols],attr_ctl,attr_reg
