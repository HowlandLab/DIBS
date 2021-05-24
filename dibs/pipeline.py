"""

For ever new pipeline implementation by the user, you must do the following:
    - Use BasicPipeline as the parent object
    - Implement `engineer_features()` using the given interface


Notes
    - the OpenTSNE implementation does not allow more than 2 components
    - GMM's "reg covar" == "regularization covariance"
TODO:
    med/high: review use of UMAP -- potential alternative to SVC?
    med/high: review use of HDBSCAN -- possible replacement for GMM clustering?
    low: implement ACTUAL random state s.t. all random state property calls beget a truly random integer
    low: review "theta"(/angle) for TSNE

Add attrib checking for engineer_features? https://duckduckgo.com/?t=ffab&q=get+all+classes+within+a+file+python&ia=web&iax=qa


"""
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle as sklearn_shuffle_dataframe

from typing import Any, Collection, Dict, List, Tuple  # TODO: med: review all uses of Optional

import os
import pandas as pd
import time

from dibs.base_pipeline import BasePipeline
from dibs.logging_enhanced import get_current_function
from dibs import check_arg, config, feature_engineering

logger = config.initialize_logger(__name__)


# Concrete pipeline implementations

class DemoPipeline(BasePipeline):
    """
    Non-functioning. Demo pipeline used for demonstration on Pipeline usage. Do not implement this into any real projects.
    """

    def engineer_features(self, data: pd.DataFrame):
        """
        Sample feature engineering function since all
        implementations of BasePipeline must implement this single function.
        """

        logger.debug(f'Engineering features for one data set...')
        logger.debug(f'Done engineering features.')
        return data


class PipelinePrime(BasePipeline):
    """
    Deprecated. First implementation of a full pipeline. Utilizes the feature set from the B-SOiD paper.
    """

    def engineer_features(self, in_df) -> pd.DataFrame:
        """

        """
        columns_to_save = ['scorer', 'source', 'file_source', 'data_source', 'frame']
        df = in_df.sort_values('frame').copy()

        # Filter
        df_filtered, _ = feature_engineering.adaptively_filter_dlc_output(df)
        # Engineer features
        df_features: pd.DataFrame = feature_engineering.engineer_7_features_dataframe(
            df_filtered, features_names_7=list(self.all_engineered_features))

        # Ensure columns don't get dropped by accident
        for col in columns_to_save:
            if col in in_df.columns and col not in df_features.columns:
                df_features[col] = df[col].values

        # Smooth over n-frame windows
        for feature in self.features_which_average_by_mean:
            df_features[feature] = feature_engineering.average_values_over_moving_window(
                df_features[feature].values, 'avg', self.average_over_n_frames)
        # Sum
        for feature in self.features_which_average_by_sum:
            df_features[feature] = feature_engineering.average_values_over_moving_window(
                df_features[feature].values, 'sum', self.average_over_n_frames)

        return df_features

    def _engineer_features_all_dfs(self, list_dfs_of_raw_data: List[pd.DataFrame]) -> pd.DataFrame:
        """
        The main function that can build features for BOTH training and prediction data.
        Here we are ensuring that the data processing for both training and prediction occurs in the same way.
        """
        # TODO: MED: these cols really should be saved in
        #  engineer_7_features_dataframe_NOMISSINGDATA(),
        #  but that func can be amended later due to time constraints

        list_dfs_raw_data = list_dfs_of_raw_data

        # Reconcile args
        if isinstance(list_dfs_raw_data, pd.DataFrame):
            list_dfs_raw_data = [list_dfs_raw_data, ]

        check_arg.ensure_type(list_dfs_raw_data, list)

        list_dfs_engineered_features: List[pd.DataFrame] = []
        for df in list_dfs_raw_data:
            df_engineered_features: pd.DataFrame = self.engineer_features(df)
            list_dfs_engineered_features.append(df_engineered_features)

        # # Adaptively filter features
        # dfs_list_adaptively_filtered: List[Tuple[pd.DataFrame, List[float]]] = [feature_engineering.adaptively_filter_dlc_output(df) for df in list_dfs_raw_data]
        #
        # # Engineer features as necessary
        # dfs_features: List[pd.DataFrame] = []
        # for df_i, _ in tqdm(dfs_list_adaptively_filtered, desc='Engineering features...'):
        #     # Save scorer, source values because the current way of engineering features strips out that info.
        #     df_features_i = feature_engineering.engineer_7_features_dataframe_NOMISSINGDATA(df_i, features_names_7=self.features_names_7)
        #     for col in columns_to_save:
        #         if col not in df_features_i.columns and col in df_i.columns:
        #             df_features_i[col] = df_i[col].values
        #     dfs_features.append(df_features_i)
        #
        # # Smooth over n-frame windows
        # for i, df in tqdm(enumerate(dfs_features), desc='Smoothing values over frames...'):
        #     # Mean
        #     for feature in self.features_which_average_by_mean:
        #         dfs_features[i][feature] = feature_engineering.average_values_over_moving_window(
        #             df[feature].values, 'avg', self.average_over_n_frames)
        #     # Sum
        #     for feature in self.features_which_average_by_sum:
        #         dfs_features[i][feature] = feature_engineering.average_values_over_moving_window(
        #             df[feature].values, 'sum', self.average_over_n_frames)

        # # Aggregate all data
        df_features = pd.concat(list_dfs_engineered_features)

        return df_features


class PipelineRetreat(BasePipeline):
    """
    Deprecated. A one-off pipeline for Tim to use at his retreat.
    """

    def engineer_features(self, in_df) -> pd.DataFrame:
        columns_to_save = ['scorer', 'source', 'file_source', 'data_source', 'frame']
        df = in_df.sort_values('frame').copy()

        # Filter
        df_filtered, _ = feature_engineering.adaptively_filter_dlc_output(df)
        # Engineer features
        df_features: pd.DataFrame = feature_engineering.engineer_7_features_dataframe(
            df_filtered,
            features_names_7=list(self.all_engineered_features),
            map_names={
                'Head': 'NOSETIP',
                'ForepawLeft': 'FOREPAW_LEFT',
                'ForepawRight': 'FOREPAW_RIGHT',
                'HindpawLeft': 'HINDPAW_LEFT',
                'HindpawRight': 'HINDPAW_RIGHT',
                'Tailbase': 'TAILBASE',
            })
        """
        TAILBASE = TailBase
        NOSETIP = NoseTip
        FOREPAW_LEFT = ForepawLeft
        FOREPAW_RIGHT = ForepawRight
        HINDPAW_LEFT = HindpawLeft
        HINDPAW_RIGHT = HindpawRight
        """
        # Ensure columns don't get dropped by accident
        for col in columns_to_save:
            if col in in_df.columns and col not in df_features.columns:
                df_features[col] = df[col].values

        # Smooth over n-frame windows
        for feature in self.features_which_average_by_mean:
            df_features[feature] = feature_engineering.average_values_over_moving_window(
                df_features[feature].values, 'avg', self.average_over_n_frames)
        # Sum
        for feature in self.features_which_average_by_sum:
            df_features[feature] = feature_engineering.average_values_over_moving_window(
                df_features[feature].values, 'sum', self.average_over_n_frames)

        return df_features


class PipelineEPM(BasePipeline):
    """
    Deprecated. First try implementation for Elevated Plus Maze whose features match those in the B-SOiD paper
    """

    def engineer_features(self, in_df) -> pd.DataFrame:
        """

        """
        check_arg.ensure_type(in_df, pd.DataFrame)
        map_mouse_point_to_config_name = {
            'Head': 'NOSETIP',
            'ForepawLeft': 'FOREPAW_LEFT',
            'ForepawRight': 'FOREPAW_RIGHT',
            'HindpawLeft': 'HINDPAW_LEFT',
            'HindpawRight': 'HINDPAW_RIGHT',
            'Tailbase': 'TAILBASE',
        }

        columns_to_save = ['scorer', 'source', 'file_source', 'data_source', 'frame']
        df = in_df.sort_values('frame').copy()

        # Filter
        df_filtered, _ = feature_engineering.adaptively_filter_dlc_output(df)

        # Engineer features
        df_features: pd.DataFrame = feature_engineering.engineer_7_features_dataframe(
            df_filtered,
            features_names_7=list(self.all_engineered_features),
            map_names=map_mouse_point_to_config_name,
        )
        # Ensure columns don't get dropped by accident
        for col in columns_to_save:
            if col in in_df.columns and col not in df_features.columns:
                df_features[col] = df[col].values

        # Smooth over n-frame windows
        for feature in self.features_which_average_by_mean:
            df_features[feature] = feature_engineering.average_values_over_moving_window(
                df_features[feature].values, 'avg', self.average_over_n_frames)
        # Sum
        for feature in self.features_which_average_by_sum:
            df_features[feature] = feature_engineering.average_values_over_moving_window(
                df_features[feature].values, 'sum', self.average_over_n_frames)

        # except Exception as e:
        #     logger.error(f'{df_features.columns} // fail on feature: {feature} // {df_features.head(10).to_string()} //{repr(e)}')
        #     raise e

        return df_features


class PipelineFlex(BasePipeline):
    """ (WIP) """
    # TODO: WIP: creating a flexible class to use with streamlit that allows for flexible feature selection

    def engineer_features(self, data: pd.DataFrame):
        # TODO
        return data


class PipelineTim(BasePipeline):
    """

    """
    # Feature names
    intermediate_avg_forepaw = 'intermediate_avg_forepaw'
    intermediate_avg_hindpaw = 'intermediate_avg_hindpaw'
    feat_name_dist_forepawleft_nosetip = 'distForepawLeftNosetip'
    feat_name_dist_forepawright_nosetip = 'distForepawRightNosetip'
    feat_name_dist_forepawLeft_hindpawLeft = 'distForepawLeftHindpawLeft'
    feat_name_dist_forepawRight_hindpawRight = 'distForepawRightHindpawRight'
    feat_name_dist_AvgHindpaw_Nosetip = 'distAvgHindpawNoseTip'
    feat_name_dist_AvgForepaw_NoseTip = 'distAvgForepawNoseTip'
    feat_name_velocity_AvgForepaw = 'velocAvgForepaw'
    _all_features = (
        feat_name_dist_forepawleft_nosetip,
        feat_name_dist_forepawright_nosetip,
        feat_name_dist_forepawLeft_hindpawLeft,
        feat_name_dist_forepawRight_hindpawRight,
        feat_name_dist_AvgHindpaw_Nosetip,
        feat_name_dist_AvgForepaw_NoseTip,
        feat_name_velocity_AvgForepaw,
    )
    # n_rows_to_integrate_by: int = 3  # 3 => 3 frames = 100ms capture in a 30fps video. # TODO: deprecate. Move property to BasePipeline

    def engineer_features(self, in_df: pd.DataFrame):
        # TODO: WIP
        """
        # Head dips
        1. d(forepaw left to nose)
        2. d(forepaw right to nose)
        # Rears
        3. d(forepaw left to hindpaw left)
        4. d(forepaw right to hindpaw right)
        5. d(nose to avg hindpaw)
        # Stretch attends
        6. d(avg hindpaw to nose) - same as #5
        7. d(avg forepaw to nose)
        8. v(avgForepaw)

        """
        # Arg Checking
        check_arg.ensure_type(in_df, pd.DataFrame)
        # Execute
        logger.debug(f'Engineering features for one data set...')
        df = in_df.sort_values('frame').copy()
        # Filter
        df, _ = feature_engineering.adaptively_filter_dlc_output(df)
        # Engineer features
        # 1
        df = feature_engineering.attach_feature_distance_between_2_bodyparts(df, 'FOREPAW_LEFT', 'NOSETIP', self.feat_name_dist_forepawleft_nosetip, resolve_bodyparts_with_config_ini=True)
        # 2
        df = feature_engineering.attach_feature_distance_between_2_bodyparts(df, 'FOREPAW_RIGHT', 'NOSETIP', self.feat_name_dist_forepawright_nosetip, resolve_bodyparts_with_config_ini=True)
        # 3
        df = feature_engineering.attach_feature_distance_between_2_bodyparts(df, 'FOREPAW_LEFT', 'HINDPAW_LEFT', self.feat_name_dist_forepawLeft_hindpawLeft, resolve_bodyparts_with_config_ini=True)
        # 4
        df = feature_engineering.attach_feature_distance_between_2_bodyparts(df, 'FOREPAW_RIGHT', 'HINDPAW_RIGHT', self.feat_name_dist_forepawRight_hindpawRight, resolve_bodyparts_with_config_ini=True)
        # 5 & 6
        # Get avg forepaw
        # df = feature_engineering.attach_average_forepaw_xy(df)  # TODO: low: replace these two functions with the generalized xy averaging functions+output name?
        df = feature_engineering.attach_average_bodypart_xy(df, 'FOREPAW_LEFT', 'FOREPAW_RIGHT', self.intermediate_avg_forepaw, resolve_bodyparts_with_config_ini=True)
        # Get avg hindpaw
        # df = feature_engineering.attach_average_hindpaw_xy(df)
        df = feature_engineering.attach_average_bodypart_xy(df, 'HINDPAW_LEFT', 'HINDPAW_RIGHT', self.intermediate_avg_hindpaw, resolve_bodyparts_with_config_ini=True)

        df = feature_engineering.attach_feature_distance_between_2_bodyparts(df, self.intermediate_avg_hindpaw, config.get_part('NOSETIP'), self.feat_name_dist_AvgHindpaw_Nosetip)
        # 7
        df = feature_engineering.attach_feature_distance_between_2_bodyparts(df, self.intermediate_avg_forepaw, config.get_part('NOSETIP'), self.feat_name_dist_AvgForepaw_NoseTip)
        # 8
        df = feature_engineering.attach_feature_velocity_of_bodypart(df, self.intermediate_avg_forepaw, 1 / config.VIDEO_FPS, output_feature_name=self.feat_name_velocity_AvgForepaw)

        # Binning
        map_feature_to_integrate_method = {
            self.feat_name_dist_forepawleft_nosetip: 'avg',
            self.feat_name_dist_forepawright_nosetip: 'avg',
            self.feat_name_dist_forepawLeft_hindpawLeft: 'avg',
            self.feat_name_dist_forepawRight_hindpawRight: 'avg',
            self.feat_name_dist_AvgHindpaw_Nosetip: 'avg',
            self.feat_name_dist_AvgForepaw_NoseTip: 'avg',
            self.feat_name_velocity_AvgForepaw: 'sum',
        }
        logger.debug(f'{get_current_function()}(): # of rows in DataFrame before binning = {len(df)}')
        df = feature_engineering.integrate_df_feature_into_bins(df, map_feature_to_integrate_method, self.average_over_n_frames)
        logger.debug(f'{get_current_function()}(): # of rows in DataFrame after binning = {len(df)}')

        # # Debug effort/check: ensure columns don't get dropped by accident
        # for col in in_df.columns:
        #     if col not in list(df.columns):
        #         err_missing_col = f'Missing col should not have been lost in feature engineering but was. ' \
        #                           f'Column = {col}. (df={df.head().to_string()})'  # TODO: low: improve err message
        #         logger.error(err_missing_col)
        #         raise KeyError(err_missing_col)

        logger.debug(f'Done engineering features.')
        return df


class PipelineCHBO(BasePipeline):
    """
    New features set created by the Howland Lab.
    """
    # Feature names
    intermediate_bodypart_avgForepaw = 'AvgForepaw'
    intermediate_bodypart_avgHindpaw = 'AvgHindpaw'
    feat_name_dist_forepawleft_nosetip = 'DistanceForepawLeftToNosetip'
    feat_name_dist_forepawright_nosetip = 'DistanceForepawRightToNosetip'
    feat_name_dist_forepawLeft_hindpawLeft = 'DistanceForepawLeftToHindpawLeft'
    feat_name_dist_forepawRight_hindpawRight = 'DistanceForepawRightToHindpawRight'
    feat_name_dist_AvgHindpaw_Nosetip = 'DistanceAvgHindpawToNosetip'
    feat_name_dist_AvgForepaw_NoseTip = 'DistanceAvgForepawToNosetip'
    feat_name_velocity_AvgForepaw = 'VelocityAvgForepaw'
    _all_features = (
        feat_name_dist_forepawleft_nosetip,
        feat_name_dist_forepawright_nosetip,
        feat_name_dist_forepawLeft_hindpawLeft,
        feat_name_dist_forepawRight_hindpawRight,
        feat_name_dist_AvgHindpaw_Nosetip,
        feat_name_dist_AvgForepaw_NoseTip,
        feat_name_velocity_AvgForepaw,
    )

    def engineer_features(self, in_df: pd.DataFrame):
        # TODO: WIP
        """
        # Head dips
        1. d(forepaw left to nose)
        2. d(forepaw right to nose)
        # Rears
        3. d(forepaw left to hindpaw left)
        4. d(forepaw right to hindpaw right)
        5. d(nose to avg hindpaw)
        # Stretch attends
        6. d(avg hindpaw to nose) - same as #5
        7. d(avg forepaw to nose)
        8. v(avgForepaw)

        """
        # Arg Checking
        check_arg.ensure_type(in_df, pd.DataFrame)
        # Execute
        logger.debug(f'Engineering features for one data set...')
        df = in_df.sort_values('frame').copy()
        # Filter
        df, _ = feature_engineering.adaptively_filter_dlc_output(df)
        # Engineer features
        # 1
        df = feature_engineering.attach_feature_distance_between_2_bodyparts(df, 'FOREPAW_LEFT', 'NOSETIP', self.feat_name_dist_forepawleft_nosetip, resolve_bodyparts_with_config_ini=True)
        # 2
        df = feature_engineering.attach_feature_distance_between_2_bodyparts(df, 'FOREPAW_RIGHT', 'NOSETIP', self.feat_name_dist_forepawright_nosetip, resolve_bodyparts_with_config_ini=True)
        # 3
        df = feature_engineering.attach_feature_distance_between_2_bodyparts(df, 'FOREPAW_LEFT', 'HINDPAW_LEFT', self.feat_name_dist_forepawLeft_hindpawLeft, resolve_bodyparts_with_config_ini=True)
        # 4
        df = feature_engineering.attach_feature_distance_between_2_bodyparts(df, 'FOREPAW_RIGHT', 'HINDPAW_RIGHT', self.feat_name_dist_forepawRight_hindpawRight, resolve_bodyparts_with_config_ini=True)
        # 5, 6
        # df = feature_engineering.attach_average_forepaw_xy(df)  # BELOW SOLVES TODO: TODO: low: replace these two functions with the generalized xy averaging functions+output name?
        df = feature_engineering.attach_average_bodypart_xy(df, 'FOREPAW_LEFT', 'FOREPAW_RIGHT', output_bodypart=self.intermediate_bodypart_avgForepaw, resolve_bodyparts_with_config_ini=True)

        # df = feature_engineering.attach_average_hindpaw_xy(df)  # BELO SOLVES TODO: low: replace these two functions with the generalized xy averaging functions+output name?
        df = feature_engineering.attach_average_bodypart_xy(df, 'HINDPAW_LEFT', 'HINDPAW_RIGHT', output_bodypart=self.intermediate_bodypart_avgHindpaw, resolve_bodyparts_with_config_ini=True)

        df = feature_engineering.attach_feature_distance_between_2_bodyparts(df, self.intermediate_bodypart_avgHindpaw, config.get_part('NOSETIP'), self.feat_name_dist_AvgHindpaw_Nosetip)

        # 7
        # df = feature_engineering.attach_distance_between_2_feats(df, 'AvgForepaw', config.get_part('NOSETIP'), self.feat_name_dist_AvgForepaw_NoseTip)
        df = feature_engineering.attach_feature_distance_between_2_bodyparts(df, self.intermediate_bodypart_avgForepaw, config.get_part('NOSETIP'), self.feat_name_dist_AvgForepaw_NoseTip)

        # 8
        # df = feature_engineering.attach_velocity_of_feature(df, 'AvgForepaw', 1/config.VIDEO_FPS, self.feat_name_velocity_AvgForepaw)
        df = feature_engineering.attach_feature_velocity_of_bodypart(df, self.intermediate_bodypart_avgForepaw, 1 / config.VIDEO_FPS, self.feat_name_velocity_AvgForepaw)

        # Binning
        map_feature_to_integrate_method = {
            self.feat_name_dist_forepawleft_nosetip: 'avg',
            self.feat_name_dist_forepawright_nosetip: 'avg',
            self.feat_name_dist_forepawLeft_hindpawLeft: 'avg',
            self.feat_name_dist_forepawRight_hindpawRight: 'avg',
            self.feat_name_dist_AvgHindpaw_Nosetip: 'avg',
            self.feat_name_dist_AvgForepaw_NoseTip: 'avg',
            self.feat_name_velocity_AvgForepaw: 'sum',
        }
        logger.debug(f'{get_current_function()}(): # of rows in DataFrame before binning = {len(df)}')
        df = feature_engineering.integrate_df_feature_into_bins(df, map_feature_to_integrate_method, self.average_over_n_frames)
        logger.debug(f'{get_current_function()}(): # of rows in DataFrame after binning = {len(df)}')

        # # Debug effort/check: ensure columns don't get dropped by accident
        # for col in in_df.columns:
        #     if col not in list(df.columns):
        #         err_missing_col = f'Missing col should not have been lost in feature engineering but was. ' \
        #                           f'Column = {col}. (df={df.head().to_string()})'  # TODO: low: improve err message
        #         logger.error(err_missing_col)
        #         raise KeyError(err_missing_col)

        logger.debug(f'Done engineering features.')
        return df


class PipelineMimic(BasePipeline):
    """
    A pipeline implementation for mimicking the B-SOID implementation.
    This is a revised version of PipelinePrime that uses a more efficient feature engineering pathway.

    7 Features listed in paper (terms in brackets are cursive and were written in math format. See paper page 12/13):

    1. body length (or "[d_ST]"): distance from snout to base of tail
    2. [d_SF]: distance of front paws to base of tail relative to body length (formally: [d_SF] = [d_ST] - [d_FT],
        where [d_FT] is the distance between front paws and base of tail
    3. [d_SB]: distance of back paws to base of tail relative to body length (formally: [d_SB] = [d_ST] - [d_BT]
    4. Inter-forepaw distance (or "[d_FP]"): the distance between the two front paws
    5. snout speed (or "[v_s]"): the displacement of the snout location over a period of 16ms
    6. base-of-tail speed (or ["v_T"]): the displacement of the base of the tail over a period of 16ms
    7. snout to base-of-tail change in angle:
    """

    # Feature names
    feat_body_length = 'BodyLength'  # 1
    intermediate_bodypart_avgForepaw = 'AvgForepaw'
    intermediate_bodypart_avgHindpaw = 'AvgHindpaw'
    intermediate_dist_avgForepaw_to_tailbase = 'DistanceAvgForepawToTailbase'
    feat_dist_front_paws_to_tailbase_relative_to_body_length = 'DistanceForepawsToTailbaseMinusBodyLength'  # 2
    intermediate_dist_avgHindpaw_to_tailbase = 'DistanceAvgHindpawToTailbase'
    feat_dist_hind_paws_to_tailbase_relative_to_body_length = 'DistanceHindpawsToTailBaseMinusBodyLength'  # 3
    feat_dist_bw_front_paws = 'DistanceBetweenForepaws'  # 4
    feat_snout_speed = 'SnoutSpeed'  # 5
    feat_tail_base_speed = 'TailSpeed'  # 6
    feat_snout_tail_delta_angle = 'DeltaAngleTailToSnout'  # 7

    _all_features = (
        feat_body_length,
        feat_dist_front_paws_to_tailbase_relative_to_body_length,
        feat_dist_hind_paws_to_tailbase_relative_to_body_length,
        feat_dist_bw_front_paws,
        feat_snout_speed,
        feat_tail_base_speed,
        feat_snout_tail_delta_angle,
    )

    def engineer_features(self, df: pd.DataFrame):
        """
            7 Features listed in paper (terms in brackets are cursive and were written in math format. See paper page 12/13):
        1. body length (or "[d_ST]"): distance from snout to base of tail
        2. [d_SF]: distance of front paws to base of tail relative to body length (formally: [d_SF] = [d_ST] - [d_FT], where [d_FT] is the distance between front paws and base of tail
        3. [d_SB]: distance of back paws to base of tail relative to body length (formally: [d_SB] = [d_ST] - [d_BT]
        4. Inter-forepaw distance (or "[d_FP]"): the distance between the two front paws

        5. snout speed (or "[v_s]"): the displacement of the snout location over a period of 16ms
        6. base-of-tail speed (or ["v_T"]): the displacement of the base of the tail over a period of 16ms
        7. snout to base-of-tail change in angle:
        """

        check_arg.ensure_type(df, pd.DataFrame)
        # Execute
        logger.debug(f'Engineering features for one data set...')
        df = df.sort_values('frame').copy()

        # 1 dist snout to tail
        df = feature_engineering.attach_feature_distance_between_2_bodyparts(df, config.get_part('TAILBASE'), config.get_part('NOSETIP'), output_feature_name=self.feat_body_length)

        # 2: Dist FrontPaws to tail relative to body length
        ## 1/3: Get AvgForepaw location
        df = feature_engineering.attach_average_bodypart_xy(df, config.get_part('FOREPAW_LEFT'), config.get_part('FOREPAW_RIGHT'), output_bodypart=self.intermediate_bodypart_avgForepaw)
        ## 2/3: Get dist from forepaw to tailbase
        df = feature_engineering.attach_feature_distance_between_2_bodyparts(df, self.intermediate_bodypart_avgForepaw, config.get_part('TAILBASE'), self.intermediate_dist_avgForepaw_to_tailbase)
        ## 3/3: Get body-length relative distance
        df[self.feat_dist_front_paws_to_tailbase_relative_to_body_length] = df[self.feat_body_length] - df[self.intermediate_dist_avgForepaw_to_tailbase]

        # 3 Dist back paws to base of tail relative to body length
        ## 1/3: Get AvgHindpaw location
        df = feature_engineering.attach_average_bodypart_xy(df, config.get_part('HINDPAW_LEFT'), config.get_part('HINDPAW_RIGHT'), output_bodypart=self.intermediate_bodypart_avgHindpaw)
        ## 2/3: Get dist from hindpaw to tailbase
        df = feature_engineering.attach_feature_distance_between_2_bodyparts(df, self.intermediate_bodypart_avgHindpaw, config.get_part('TAILBASE'), output_feature_name=self.intermediate_dist_avgHindpaw_to_tailbase)
        ## 3/3: Get body-length relative distance
        df[self.feat_dist_hind_paws_to_tailbase_relative_to_body_length] = df[self.feat_body_length] - df[self.intermediate_dist_avgHindpaw_to_tailbase]

        # 4: distance between 2 front paws
        df = feature_engineering.attach_feature_distance_between_2_bodyparts(df, config.get_part('FOREPAW_LEFT'), config.get_part('FOREPAW_RIGHT'), self.feat_dist_bw_front_paws)

        # 5: snout speed
        df = feature_engineering.attach_feature_velocity_of_bodypart(df, config.get_part('NOSETIP'), action_duration=1 / self.video_fps, output_feature_name=self.feat_snout_speed)

        # 6 tail speed
        df = feature_engineering.attach_feature_velocity_of_bodypart(df, config.get_part('TAILBASE'), action_duration=1 / self.video_fps, output_feature_name=self.feat_tail_base_speed)

        # 7: snout to base of tail change in angle
        df = feature_engineering.attach_angle_between_bodyparts(df, config.get_part('NOSETIP'), config.get_part('TAILBASE'), self.feat_snout_tail_delta_angle)

        # BINNING #
        map_feature_to_integrate_method = {
            self.feat_body_length: 'avg',
            self.feat_dist_front_paws_to_tailbase_relative_to_body_length: 'avg',
            self.feat_dist_hind_paws_to_tailbase_relative_to_body_length: 'avg',
            self.feat_dist_bw_front_paws: 'avg',
            self.feat_snout_speed: 'sum',
            self.feat_tail_base_speed: 'sum',
            self.feat_snout_tail_delta_angle: 'sum',
        }

        logger.debug(f'{get_current_function()}(): # of rows in DataFrame before binning = {len(df)}')
        df = feature_engineering.integrate_df_feature_into_bins(df, map_feature_to_integrate_method, self.average_over_n_frames)
        logger.debug(f'{get_current_function()}(): # of rows in DataFrame after binning = {len(df)}')

        return df


class PipelineHowland(BasePipeline):
    """
    New features set created by the Howland Lab.
    """
    # Feature names
    intermediate_bodypart_avgForepaw = 'AvgForepaw'
    intermediate_bodypart_avgHindpaw = 'AvgHindpaw'
    feat_name_dist_forepawleft_nosetip = 'DistanceForepawLeftToNosetip'
    feat_name_dist_forepawright_nosetip = 'DistanceForepawRightToNosetip'
    feat_name_dist_forepawLeft_hindpawLeft = 'DistanceForepawLeftToHindpawLeft'
    feat_name_dist_forepawRight_hindpawRight = 'DistanceForepawRightToHindpawRight'
    feat_name_dist_AvgHindpaw_Nosetip = 'DistanceAvgHindpawToNosetip'
    feat_name_dist_AvgForepaw_NoseTip = 'DistanceAvgForepawToNosetip'
    feat_name_velocity_AvgForepaw = 'VelocityAvgForepaw'
    _all_features = (
        feat_name_dist_forepawleft_nosetip,
        feat_name_dist_forepawright_nosetip,
        feat_name_dist_forepawLeft_hindpawLeft,
        feat_name_dist_forepawRight_hindpawRight,
        feat_name_dist_AvgHindpaw_Nosetip,
        feat_name_dist_AvgForepaw_NoseTip,
        feat_name_velocity_AvgForepaw,
    )

    def engineer_features(self, in_df: pd.DataFrame):
        # TODO: WIP
        """
        # Head dips
        1. d(forepaw left to nose)
        2. d(forepaw right to nose)
        # Rears
        3. d(forepaw left to hindpaw left)
        4. d(forepaw right to hindpaw right)
        5. d(nose to avg hindpaw)
        # Stretch attends
        6. d(avg hindpaw to nose) - same as #5
        7. d(avg forepaw to nose)
        8. v(avgForepaw)

        """
        # Arg Checking
        check_arg.ensure_type(in_df, pd.DataFrame)
        # Execute
        logger.debug(f'Engineering features for one data set...')
        df = in_df.astype({'frame': int}).sort_values('frame').copy()
        # Filter
        df, _ = feature_engineering.adaptively_filter_dlc_output(df)
        # Engineer features
        # 1
        df = feature_engineering.attach_feature_distance_between_2_bodyparts(df, config.get_part('FOREPAW_LEFT'), config.get_part('NOSETIP'), self.feat_name_dist_forepawleft_nosetip)
        # 2
        df = feature_engineering.attach_feature_distance_between_2_bodyparts(df, config.get_part('FOREPAW_RIGHT'), config.get_part('NOSETIP'), self.feat_name_dist_forepawright_nosetip)
        # 3
        df = feature_engineering.attach_feature_distance_between_2_bodyparts(df, config.get_part('FOREPAW_LEFT'), config.get_part('HINDPAW_LEFT'), self.feat_name_dist_forepawLeft_hindpawLeft)
        # 4
        df = feature_engineering.attach_feature_distance_between_2_bodyparts(df, config.get_part('FOREPAW_RIGHT'), config.get_part('HINDPAW_RIGHT'), self.feat_name_dist_forepawRight_hindpawRight)
        # 5, 6
        df = feature_engineering.attach_average_bodypart_xy(df, config.get_part('FOREPAW_LEFT'), config.get_part('FOREPAW_RIGHT'), output_bodypart=self.intermediate_bodypart_avgForepaw)
        df = feature_engineering.attach_average_bodypart_xy(df, config.get_part('HINDPAW_LEFT'), config.get_part('HINDPAW_RIGHT'), output_bodypart=self.intermediate_bodypart_avgHindpaw)
        df = feature_engineering.attach_feature_distance_between_2_bodyparts(df, self.intermediate_bodypart_avgHindpaw, config.get_part('NOSETIP'), self.feat_name_dist_AvgHindpaw_Nosetip)

        # 7
        df = feature_engineering.attach_feature_distance_between_2_bodyparts(df, self.intermediate_bodypart_avgForepaw, config.get_part('NOSETIP'), output_feature_name=self.feat_name_dist_AvgForepaw_NoseTip)

        # 8
        # df = feature_engineering.attach_velocity_of_feature(df, 'AvgForepaw', 1/config.VIDEO_FPS, self.feat_name_velocity_AvgForepaw)
        df = feature_engineering.attach_feature_velocity_of_bodypart(df, self.intermediate_bodypart_avgForepaw, action_duration=1 / config.VIDEO_FPS, output_feature_name=self.feat_name_velocity_AvgForepaw)

        # Binning
        map_feature_to_integrate_method = {
            self.feat_name_dist_forepawleft_nosetip: 'avg',
            self.feat_name_dist_forepawright_nosetip: 'avg',
            self.feat_name_dist_forepawLeft_hindpawLeft: 'avg',
            self.feat_name_dist_forepawRight_hindpawRight: 'avg',
            self.feat_name_dist_AvgHindpaw_Nosetip: 'avg',
            self.feat_name_dist_AvgForepaw_NoseTip: 'avg',
            self.feat_name_velocity_AvgForepaw: 'sum',
        }
        logger.debug(f'{get_current_function()}(): # of rows in DataFrame before binning = {len(df)}')
        df = feature_engineering.integrate_df_feature_into_bins(df, map_feature_to_integrate_method, self.average_over_n_frames)
        logger.debug(f'{get_current_function()}(): # of rows in DataFrame after binning = {len(df)}')

        # # Debug effort/check: ensure columns don't get dropped by accident
        # for col in in_df.columns:
        #     if col not in list(df.columns):
        #         err_missing_col = f'Missing col should not have been lost in feature engineering but was. ' \
        #                           f'Column = {col}. (df={df.head().to_string()})'  # TODO: low: improve err message
        #         logger.error(err_missing_col)
        #         raise KeyError(err_missing_col)

        logger.debug(f'Done engineering features.')
        return df


class PipelineHowlandUMAP(PipelineHowland):

    def _train_tsne_get_dimension_reduced_data(self, data):
        logger.debug(f'Now logging with PIPELINESTANDIN with UMAP')
        reducer = umap.UMAP(

            n_neighbors=self.umap_n_neighbors,
            n_components=self.tsne_n_components,
            learning_rate=self.umap_learning_rate,
            n_jobs=self.tsne_n_jobs,
            low_memory=False,
        )

        arr_result = reducer.fit_transform(data[list(self.all_engineered_features)].values)
        return arr_result


class PipelineHowlandLLE(PipelineHowland):
    def _train_tsne_get_dimension_reduced_data(self, data):
        logger.debug(f'Now logging with PIPELINESTANDIN with LLE')

        reducer = LocallyLinearEmbedding(
            method=self.LLE_method,

            n_neighbors=self.LLE_n_neighbors,
            n_components=self.tsne_n_components,
            reg=1E-3,
            eigen_solver='auto', tol=1E-6, max_iter=100,
            hessian_tol=1E-4, modified_tol=1E-12,
            neighbors_algorithm='auto',

            random_state=self.random_state,
            n_jobs=self.tsne_n_jobs,
        )

        arr_result = reducer.fit_transform(data[list(self.all_engineered_features)].values)
        return arr_result


class PipelineKitchenSink(BasePipeline):
    """
    New features set created by the Howland Lab.
    """
    # Feature names

    intermediate_bodypart_avgForepaw = 'AvgForepaw'
    intermediate_bodypart_avgHindpaw = 'AvgHindpaw'
    feat_name_dist_forepawleft_nosetip = 'DistanceForepawLeftToNosetip'
    feat_name_dist_forepawright_nosetip = 'DistanceForepawRightToNosetip'
    feat_name_dist_forepawLeft_hindpawLeft = 'DistanceForepawLeftToHindpawLeft'
    feat_name_dist_forepawRight_hindpawRight = 'DistanceForepawRightToHindpawRight'
    feat_name_dist_AvgHindpaw_Nosetip = 'DistanceAvgHindpawToNosetip'
    feat_name_dist_AvgForepaw_NoseTip = 'DistanceAvgForepawToNosetip'
    feat_name_velocity_AvgForepaw = 'VelocityAvgForepaw'

    feat_body_length = 'BodyLength'  # 1
    intermediate_dist_avgForepaw_to_tailbase = 'DistanceAvgForepawToTailbase'
    feat_dist_front_paws_to_tailbase_relative_to_body_length = 'DistanceForepawsToTailbaseMinusBodyLength'  # 2
    intermediate_dist_avgHindpaw_to_tailbase = 'DistanceAvgHindpawToTailbase'
    feat_dist_hind_paws_to_tailbase_relative_to_body_length = 'DistanceHindpawsToTailBaseMinusBodyLength'  # 3
    feat_dist_bw_front_paws = 'DistanceBetweenForepaws'  # 4
    feat_snout_speed = 'SnoutSpeed'  # 5
    feat_tail_base_speed = 'TailSpeed'  # 6
    feat_snout_tail_delta_angle = 'DeltaAngleTailToSnout'  # 7

    _all_features = (
        feat_name_dist_forepawleft_nosetip,
        feat_name_dist_forepawright_nosetip,
        feat_name_dist_forepawLeft_hindpawLeft,
        feat_name_dist_forepawRight_hindpawRight,
        feat_name_dist_AvgHindpaw_Nosetip,
        feat_name_dist_AvgForepaw_NoseTip,
        feat_name_velocity_AvgForepaw,
        feat_body_length,
        intermediate_dist_avgForepaw_to_tailbase,  # 1
        feat_dist_front_paws_to_tailbase_relative_to_body_length,  # 2
        intermediate_dist_avgHindpaw_to_tailbase,
        feat_dist_hind_paws_to_tailbase_relative_to_body_length,  # 3
        feat_dist_bw_front_paws,  # 4
        feat_snout_speed,  # 5
        feat_tail_base_speed,  # 6
        feat_snout_tail_delta_angle,  # 7
    )

    def engineer_features(self, in_df: pd.DataFrame):
        # TODO: WIP
        """
        # Head dips
        1. d(forepaw left to nose)
        2. d(forepaw right to nose)
        # Rears
        3. d(forepaw left to hindpaw left)
        4. d(forepaw right to hindpaw right)
        5. d(nose to avg hindpaw)
        # Stretch attends
        6. d(avg hindpaw to nose) - same as #5
        7. d(avg forepaw to nose)
        8. v(avgForepaw)

        """
        # Arg Checking
        check_arg.ensure_type(in_df, pd.DataFrame)
        # Execute
        logger.debug(f'Engineering features for one data set...')
        df = in_df.sort_values('frame').copy()
        # Filter
        df, _ = feature_engineering.adaptively_filter_dlc_output(df)
        # Engineer features
        # 1
        df = feature_engineering.attach_feature_distance_between_2_bodyparts(df, config.get_part('FOREPAW_LEFT'), config.get_part('NOSETIP'), self.feat_name_dist_forepawleft_nosetip)
        # 2
        df = feature_engineering.attach_feature_distance_between_2_bodyparts(df, config.get_part('FOREPAW_RIGHT'), config.get_part('NOSETIP'), self.feat_name_dist_forepawright_nosetip)
        # 3
        df = feature_engineering.attach_feature_distance_between_2_bodyparts(df, config.get_part('FOREPAW_LEFT'), config.get_part('HINDPAW_LEFT'), self.feat_name_dist_forepawLeft_hindpawLeft)
        # 4
        df = feature_engineering.attach_feature_distance_between_2_bodyparts(df, config.get_part('FOREPAW_RIGHT'), config.get_part('HINDPAW_RIGHT'), self.feat_name_dist_forepawRight_hindpawRight)
        # 5, 6
        # df = feature_engineering.attach_average_forepaw_xy(df)  # BELOW SOLVES TODO: TODO: low: replace these two functions with the generalized xy averaging functions+output name?
        df = feature_engineering.attach_average_bodypart_xy(df, config.get_part('FOREPAW_LEFT'), config.get_part('FOREPAW_RIGHT'), output_bodypart=self.intermediate_bodypart_avgForepaw)

        # df = feature_engineering.attach_average_hindpaw_xy(df)  # BELO SOLVES TODO: low: replace these two functions with the generalized xy averaging functions+output name?
        df = feature_engineering.attach_average_bodypart_xy(df, config.get_part('HINDPAW_LEFT'), config.get_part('HINDPAW_RIGHT'), output_bodypart=self.intermediate_bodypart_avgHindpaw)

        df = feature_engineering.attach_feature_distance_between_2_bodyparts(df, self.intermediate_bodypart_avgHindpaw, config.get_part('NOSETIP'), self.feat_name_dist_AvgHindpaw_Nosetip)

        # 7
        # df = feature_engineering.attach_distance_between_2_feats(df, 'AvgForepaw', config.get_part('NOSETIP'), self.feat_name_dist_AvgForepaw_NoseTip)
        df = feature_engineering.attach_feature_distance_between_2_bodyparts(df, self.intermediate_bodypart_avgForepaw, config.get_part('NOSETIP'), self.feat_name_dist_AvgForepaw_NoseTip)

        # 8
        # df = feature_engineering.attach_velocity_of_feature(df, 'AvgForepaw', 1/config.VIDEO_FPS, self.feat_name_velocity_AvgForepaw)
        df = feature_engineering.attach_feature_velocity_of_bodypart(df, self.intermediate_bodypart_avgForepaw, action_duration=1 / config.VIDEO_FPS, output_feature_name=self.feat_name_velocity_AvgForepaw)

        #####################################################
        # 1 dist snout to tail
        df = feature_engineering.attach_feature_distance_between_2_bodyparts(df, config.get_part('TAILBASE'), config.get_part('NOSETIP'), output_feature_name=self.feat_body_length)

        # 2: Dist FrontPaws to tail relative to body length
        ## 1/3: Get AvgForepaw location
        df = feature_engineering.attach_average_bodypart_xy(df, config.get_part('FOREPAW_LEFT'), config.get_part('FOREPAW_RIGHT'), output_bodypart=self.intermediate_bodypart_avgForepaw)
        ## 2/3: Get dist from forepaw to tailbase
        df = feature_engineering.attach_feature_distance_between_2_bodyparts(df, self.intermediate_bodypart_avgForepaw, config.get_part('TAILBASE'), self.intermediate_dist_avgForepaw_to_tailbase)
        ## 3/3: Get body-length relative distance
        df[self.feat_dist_front_paws_to_tailbase_relative_to_body_length] = df[self.feat_body_length] - df[self.intermediate_dist_avgForepaw_to_tailbase]

        # 3 Dist back paws to base of tail relative to body length
        ## 1/3: Get AvgHindpaw location
        df = feature_engineering.attach_average_bodypart_xy(df, config.get_part('HINDPAW_LEFT'), config.get_part('HINDPAW_RIGHT'), output_bodypart=self.intermediate_bodypart_avgHindpaw)
        ## 2/3: Get dist from hindpaw to tailbase
        df = feature_engineering.attach_feature_distance_between_2_bodyparts(df, self.intermediate_bodypart_avgHindpaw, config.get_part('TAILBASE'), output_feature_name=self.intermediate_dist_avgHindpaw_to_tailbase)
        ## 3/3: Get body-length relative distance
        df[self.feat_dist_hind_paws_to_tailbase_relative_to_body_length] = df[self.feat_body_length] - df[self.intermediate_dist_avgHindpaw_to_tailbase]

        # 4: distance between 2 front paws
        df = feature_engineering.attach_feature_distance_between_2_bodyparts(df, config.get_part('FOREPAW_LEFT'), config.get_part('FOREPAW_RIGHT'), self.feat_dist_bw_front_paws)

        # 5: snout speed
        df = feature_engineering.attach_feature_velocity_of_bodypart(df, config.get_part('NOSETIP'), action_duration=1 / self.video_fps, output_feature_name=self.feat_snout_speed)

        # 6 tail speed
        df = feature_engineering.attach_feature_velocity_of_bodypart(df, config.get_part('TAILBASE'), action_duration=1 / self.video_fps, output_feature_name=self.feat_tail_base_speed)

        # 7: snout to base of tail change in angle
        df = feature_engineering.attach_angle_between_bodyparts(df, config.get_part('NOSETIP'), config.get_part('TAILBASE'), self.feat_snout_tail_delta_angle)

        # BINNING #
        map_feature_to_integrate_method = {
            self.feat_name_dist_forepawleft_nosetip: 'avg',
            self.feat_name_dist_forepawright_nosetip: 'avg',
            self.feat_name_dist_forepawLeft_hindpawLeft: 'avg',
            self.feat_name_dist_forepawRight_hindpawRight: 'avg',
            self.feat_name_dist_AvgHindpaw_Nosetip: 'avg',
            self.feat_name_dist_AvgForepaw_NoseTip: 'avg',
            self.feat_name_velocity_AvgForepaw: 'sum',

            self.feat_body_length: 'avg',
            self.feat_dist_front_paws_to_tailbase_relative_to_body_length: 'avg',
            self.feat_dist_hind_paws_to_tailbase_relative_to_body_length: 'avg',
            self.feat_dist_bw_front_paws: 'avg',
            self.feat_snout_speed: 'sum',
            self.feat_tail_base_speed: 'sum',
            self.feat_snout_tail_delta_angle: 'sum',
        }
        logger.debug(f'{get_current_function()}(): # of rows in DataFrame before binning = {len(df)}')
        df = feature_engineering.integrate_df_feature_into_bins(df, map_feature_to_integrate_method, self.average_over_n_frames)
        logger.debug(f'{get_current_function()}(): # of rows in DataFrame after binning = {len(df)}')

        # # Debug effort/check: ensure columns don't get dropped by accident
        # for col in in_df.columns:
        #     if col not in list(df.columns):
        #         err_missing_col = f'Missing col should not have been lost in feature engineering but was. ' \
        #                           f'Column = {col}. (df={df.head().to_string()})'  # TODO: low: improve err message
        #         logger.error(err_missing_col)
        #         raise KeyError(err_missing_col)

        logger.debug(f'Done engineering features.')
        return df


class PipelineIBNS_first_one(BasePipeline):
    """
    New features set created by the Howland Lab.
    """
    # Feature names

    intermediate_bodypart_avgForepaw = 'AvgForepaw'
    intermediate_bodypart_avgHindpaw = 'AvgHindpaw'
    feat_name_dist_forepawleft_nosetip = 'DistanceForepawLeftToNosetip'
    feat_name_dist_forepawright_nosetip = 'DistanceForepawRightToNosetip'
    feat_name_dist_forepawLeft_hindpawLeft = 'DistanceForepawLeftToHindpawLeft'
    feat_name_dist_forepawRight_hindpawRight = 'DistanceForepawRightToHindpawRight'
    feat_name_dist_AvgHindpaw_Nosetip = 'DistanceAvgHindpawToNosetip'
    feat_name_dist_AvgForepaw_NoseTip = 'DistanceAvgForepawToNosetip'
    feat_name_velocity_AvgForepaw = 'VelocityAvgForepaw'

    feat_body_length = 'BodyLength'  # 1
    intermediate_dist_avgForepaw_to_tailbase = 'DistanceAvgForepawToTailbase'
    feat_dist_front_paws_to_tailbase_relative_to_body_length = 'DistanceForepawsToTailbaseMinusBodyLength'  # 2
    intermediate_dist_avgHindpaw_to_tailbase = 'DistanceAvgHindpawToTailbase'
    feat_dist_hind_paws_to_tailbase_relative_to_body_length = 'DistanceHindpawsToTailBaseMinusBodyLength'  # 3
    feat_dist_bw_front_paws = 'DistanceBetweenForepaws'  # 4
    feat_snout_speed = 'SnoutSpeed'  # 5
    feat_tail_base_speed = 'TailSpeed'  # 6
    feat_snout_tail_delta_angle = 'DeltaAngleTailToSnout'  # 7

    feat_dist_nose_obj1 = 'DistanceNosetipToObj1'
    feat_velocity_nose_obj1 = 'VelocityNosetipToObj1'
    feat_dist_nose_obj2 = 'DistanceNosetipToObj2'
    feat_velocity_nose_obj2 = 'VelocityNosetipToObj2'
    feat_dist_nose_obj3 = 'DistanceNosetipToObj3'
    feat_velocity_nose_obj3 = 'VelocityNosetipToObj3'
    feat_dist_nose_obj4 = 'DistanceNosetipToObj4'
    feat_velocity_nose_obj4 = 'VelocityNosetipToObj4'
    feat_dist_nose_obj5 = 'DistanceNosetipToObj5'
    feat_velocity_nose_obj5 = 'VelocityNosetipToObj5'
    feat_dist_nose_obj6 = 'DistanceNosetipToObj6'
    feat_velocity_nose_obj6 = 'VelocityNosetipToObj6'

    _all_features = (
        feat_name_dist_forepawleft_nosetip,
        feat_name_dist_forepawright_nosetip,
        feat_name_dist_forepawLeft_hindpawLeft,
        feat_name_dist_forepawRight_hindpawRight,
        feat_name_dist_AvgHindpaw_Nosetip,
        feat_name_dist_AvgForepaw_NoseTip,
        feat_name_velocity_AvgForepaw,
        feat_body_length,
        intermediate_dist_avgForepaw_to_tailbase,  # 1
        feat_dist_front_paws_to_tailbase_relative_to_body_length,  # 2
        intermediate_dist_avgHindpaw_to_tailbase,
        feat_dist_hind_paws_to_tailbase_relative_to_body_length,  # 3
        feat_dist_bw_front_paws,  # 4
        feat_snout_speed,  # 5
        feat_tail_base_speed,  # 6
        feat_snout_tail_delta_angle,  # 7
        feat_dist_nose_obj1,
        feat_velocity_nose_obj1,
        feat_dist_nose_obj2,
        feat_velocity_nose_obj2,
        feat_dist_nose_obj3,
        feat_velocity_nose_obj3,
        feat_dist_nose_obj4,
        feat_velocity_nose_obj4,
        feat_dist_nose_obj5,
        feat_velocity_nose_obj5,
        feat_dist_nose_obj6,
        feat_velocity_nose_obj6,
    )

    def engineer_features(self, in_df: pd.DataFrame):
        # TODO: WIP
        """
        # Head dips
        1. d(forepaw left to nose)
        2. d(forepaw right to nose)
        # Rears
        3. d(forepaw left to hindpaw left)
        4. d(forepaw right to hindpaw right)
        5. d(nose to avg hindpaw)
        # Stretch attends
        6. d(avg hindpaw to nose) - same as #5
        7. d(avg forepaw to nose)
        8. v(avgForepaw)

        """
        # Arg Checking
        check_arg.ensure_type(in_df, pd.DataFrame)
        expected_cols = ['Object1_y', 'Object2_x']
        check_arg.ensure_columns_in_DataFrame(in_df, expected_cols)
        # Execute
        logger.debug(f'Engineering features for one data set...')
        df = in_df.sort_values('frame').copy()
        # Filter
        df, _ = feature_engineering.adaptively_filter_dlc_output(df)
        # Engineer features
        # 1
        df = feature_engineering.attach_feature_distance_between_2_bodyparts(df, config.get_part('FOREPAW_LEFT'), config.get_part('NOSETIP'), self.feat_name_dist_forepawleft_nosetip)
        # 2
        df = feature_engineering.attach_feature_distance_between_2_bodyparts(df, config.get_part('FOREPAW_RIGHT'), config.get_part('NOSETIP'), self.feat_name_dist_forepawright_nosetip)
        # 3
        df = feature_engineering.attach_feature_distance_between_2_bodyparts(df, config.get_part('FOREPAW_LEFT'), config.get_part('HINDPAW_LEFT'), self.feat_name_dist_forepawLeft_hindpawLeft)
        # 4
        df = feature_engineering.attach_feature_distance_between_2_bodyparts(df, config.get_part('FOREPAW_RIGHT'), config.get_part('HINDPAW_RIGHT'), self.feat_name_dist_forepawRight_hindpawRight)
        # 5, 6
        # df = feature_engineering.attach_average_forepaw_xy(df)  # BELOW SOLVES TODO: TODO: low: replace these two functions with the generalized xy averaging functions+output name?
        df = feature_engineering.attach_average_bodypart_xy(df, config.get_part('FOREPAW_LEFT'), config.get_part('FOREPAW_RIGHT'), output_bodypart=self.intermediate_bodypart_avgForepaw)

        # df = feature_engineering.attach_average_hindpaw_xy(df)  # BELO SOLVES TODO: low: replace these two functions with the generalized xy averaging functions+output name?
        df = feature_engineering.attach_average_bodypart_xy(df, config.get_part('HINDPAW_LEFT'), config.get_part('HINDPAW_RIGHT'), output_bodypart=self.intermediate_bodypart_avgHindpaw)

        df = feature_engineering.attach_feature_distance_between_2_bodyparts(df, self.intermediate_bodypart_avgHindpaw, config.get_part('NOSETIP'), self.feat_name_dist_AvgHindpaw_Nosetip)

        # 7
        # df = feature_engineering.attach_distance_between_2_feats(df, 'AvgForepaw', config.get_part('NOSETIP'), self.feat_name_dist_AvgForepaw_NoseTip)
        df = feature_engineering.attach_feature_distance_between_2_bodyparts(df, self.intermediate_bodypart_avgForepaw, config.get_part('NOSETIP'), self.feat_name_dist_AvgForepaw_NoseTip)

        # 8
        # df = feature_engineering.attach_velocity_of_feature(df, 'AvgForepaw', 1/config.VIDEO_FPS, self.feat_name_velocity_AvgForepaw)
        df = feature_engineering.attach_feature_velocity_of_bodypart(df, self.intermediate_bodypart_avgForepaw, action_duration=1 / config.VIDEO_FPS, output_feature_name=self.feat_name_velocity_AvgForepaw)

        #####################################################
        # 1 dist snout to tail
        df = feature_engineering.attach_feature_distance_between_2_bodyparts(df, config.get_part('TAILBASE'), config.get_part('NOSETIP'), output_feature_name=self.feat_body_length)

        # 2: Dist FrontPaws to tail relative to body length
        ## 1/3: Get AvgForepaw location
        df = feature_engineering.attach_average_bodypart_xy(df, config.get_part('FOREPAW_LEFT'), config.get_part('FOREPAW_RIGHT'), output_bodypart=self.intermediate_bodypart_avgForepaw)
        ## 2/3: Get dist from forepaw to tailbase
        df = feature_engineering.attach_feature_distance_between_2_bodyparts(df, self.intermediate_bodypart_avgForepaw, config.get_part('TAILBASE'), self.intermediate_dist_avgForepaw_to_tailbase)
        ## 3/3: Get body-length relative distance
        df[self.feat_dist_front_paws_to_tailbase_relative_to_body_length] = df[self.feat_body_length] - df[self.intermediate_dist_avgForepaw_to_tailbase]

        # 3 Dist back paws to base of tail relative to body length
        ## 1/3: Get AvgHindpaw location
        df = feature_engineering.attach_average_bodypart_xy(df, config.get_part('HINDPAW_LEFT'), config.get_part('HINDPAW_RIGHT'), output_bodypart=self.intermediate_bodypart_avgHindpaw)
        ## 2/3: Get dist from hindpaw to tailbase
        df = feature_engineering.attach_feature_distance_between_2_bodyparts(df, self.intermediate_bodypart_avgHindpaw, config.get_part('TAILBASE'), output_feature_name=self.intermediate_dist_avgHindpaw_to_tailbase)
        ## 3/3: Get body-length relative distance
        df[self.feat_dist_hind_paws_to_tailbase_relative_to_body_length] = df[self.feat_body_length] - df[self.intermediate_dist_avgHindpaw_to_tailbase]

        # 4: distance between 2 front paws
        df = feature_engineering.attach_feature_distance_between_2_bodyparts(df, config.get_part('FOREPAW_LEFT'), config.get_part('FOREPAW_RIGHT'), self.feat_dist_bw_front_paws)

        # 5: snout speed
        df = feature_engineering.attach_feature_velocity_of_bodypart(df, config.get_part('NOSETIP'), action_duration=1 / self.video_fps, output_feature_name=self.feat_snout_speed)

        # 6 tail speed
        df = feature_engineering.attach_feature_velocity_of_bodypart(df, config.get_part('TAILBASE'), action_duration=1 / self.video_fps, output_feature_name=self.feat_tail_base_speed)

        # 7: snout to base of tail change in angle
        df = feature_engineering.attach_angle_between_bodyparts(df, config.get_part('NOSETIP'), config.get_part('TAILBASE'), self.feat_snout_tail_delta_angle)

        logger.warn(f'COLS: {df.columns}')

        df = feature_engineering.attach_feature_distance_between_2_bodyparts(df, config.get_part('NOSETIP'), 'Object1', output_feature_name=self.feat_dist_nose_obj1)
        df = feature_engineering.attach_feature_distance_between_2_bodyparts(df, config.get_part('NOSETIP'), 'Object2', output_feature_name=self.feat_dist_nose_obj2)
        df = feature_engineering.attach_feature_distance_between_2_bodyparts(df, config.get_part('NOSETIP'), 'Object3', output_feature_name=self.feat_dist_nose_obj3)
        df = feature_engineering.attach_feature_distance_between_2_bodyparts(df, config.get_part('NOSETIP'), 'Object4', output_feature_name=self.feat_dist_nose_obj4)
        df = feature_engineering.attach_feature_distance_between_2_bodyparts(df, config.get_part('NOSETIP'), 'Object5', output_feature_name=self.feat_dist_nose_obj5)
        df = feature_engineering.attach_feature_distance_between_2_bodyparts(df, config.get_part('NOSETIP'), 'Object6', output_feature_name=self.feat_dist_nose_obj6)
        # DistanceNosetipToObj1_x
        df = feature_engineering.attach_delta_of_single_column(df, self.feat_dist_nose_obj1, action_duration=1/self.video_fps, output_feature_name=self.feat_velocity_nose_obj1)
        df = feature_engineering.attach_delta_of_single_column(df, self.feat_dist_nose_obj2, action_duration=1/self.video_fps, output_feature_name=self.feat_velocity_nose_obj2)
        df = feature_engineering.attach_delta_of_single_column(df, self.feat_dist_nose_obj3, action_duration=1/self.video_fps, output_feature_name=self.feat_velocity_nose_obj3)
        df = feature_engineering.attach_delta_of_single_column(df, self.feat_dist_nose_obj4, action_duration=1/self.video_fps, output_feature_name=self.feat_velocity_nose_obj4)
        df = feature_engineering.attach_delta_of_single_column(df, self.feat_dist_nose_obj5, action_duration=1/self.video_fps, output_feature_name=self.feat_velocity_nose_obj5)
        df = feature_engineering.attach_delta_of_single_column(df, self.feat_dist_nose_obj6, action_duration=1/self.video_fps, output_feature_name=self.feat_velocity_nose_obj6)

        # BINNING #
        map_feature_to_integrate_method = {
            self.feat_name_dist_forepawleft_nosetip: 'avg',
            self.feat_name_dist_forepawright_nosetip: 'avg',
            self.feat_name_dist_forepawLeft_hindpawLeft: 'avg',
            self.feat_name_dist_forepawRight_hindpawRight: 'avg',
            self.feat_name_dist_AvgHindpaw_Nosetip: 'avg',
            self.feat_name_dist_AvgForepaw_NoseTip: 'avg',
            self.feat_name_velocity_AvgForepaw: 'sum',

            self.feat_body_length: 'avg',
            self.feat_dist_front_paws_to_tailbase_relative_to_body_length: 'avg',
            self.feat_dist_hind_paws_to_tailbase_relative_to_body_length: 'avg',
            self.feat_dist_bw_front_paws: 'avg',
            self.feat_snout_speed: 'sum',
            self.feat_tail_base_speed: 'sum',
            self.feat_snout_tail_delta_angle: 'sum',

            self.feat_dist_nose_obj1: 'avg',
            self.feat_velocity_nose_obj1: 'avg',
            self.feat_dist_nose_obj2: 'avg',
            self.feat_velocity_nose_obj2: 'avg',
            self.feat_dist_nose_obj3: 'avg',
            self.feat_velocity_nose_obj3: 'avg',
            self.feat_dist_nose_obj4: 'avg',
            self.feat_velocity_nose_obj4: 'avg',
            self.feat_dist_nose_obj5: 'avg',
            self.feat_velocity_nose_obj5: 'avg',
            self.feat_dist_nose_obj6: 'avg',
            self.feat_velocity_nose_obj6: 'avg',
        }
        logger.debug(f'{get_current_function()}(): # of rows in DataFrame before binning = {len(df)}')
        df = feature_engineering.integrate_df_feature_into_bins(df, map_feature_to_integrate_method, self.average_over_n_frames)
        logger.debug(f'{get_current_function()}(): # of rows in DataFrame after binning = {len(df)}')

        # # Debug effort/check: ensure columns don't get dropped by accident
        # for col in in_df.columns:
        #     if col not in list(df.columns):
        #         err_missing_col = f'Missing col should not have been lost in feature engineering but was. ' \
        #                           f'Column = {col}. (df={df.head().to_string()})'  # TODO: low: improve err message
        #         logger.error(err_missing_col)
        #         raise KeyError(err_missing_col)

        logger.debug(f'Done engineering features.')
        return df


class PipelineIBNS_second_with_time_shifting(BasePipeline):
    """
    New features set created by the Howland Lab.
    """
    # Feature names

    intermediate_bodypart_avgForepaw = 'AvgForepaw'
    intermediate_bodypart_avgHindpaw = 'AvgHindpaw'
    feat_name_dist_forepawleft_nosetip = 'DistanceForepawLeftToNosetip'
    feat_name_dist_forepawright_nosetip = 'DistanceForepawRightToNosetip'
    feat_name_dist_forepawLeft_hindpawLeft = 'DistanceForepawLeftToHindpawLeft'
    feat_name_dist_forepawRight_hindpawRight = 'DistanceForepawRightToHindpawRight'
    feat_name_dist_AvgHindpaw_Nosetip = 'DistanceAvgHindpawToNosetip'
    feat_name_dist_AvgForepaw_NoseTip = 'DistanceAvgForepawToNosetip'
    feat_name_velocity_AvgForepaw = 'VelocityAvgForepaw'

    feat_body_length = 'BodyLength'  # 1
    intermediate_dist_avgForepaw_to_tailbase = 'DistanceAvgForepawToTailbase'
    feat_dist_front_paws_to_tailbase_relative_to_body_length = 'DistanceForepawsToTailbaseMinusBodyLength'  # 2
    intermediate_dist_avgHindpaw_to_tailbase = 'DistanceAvgHindpawToTailbase'
    feat_dist_hind_paws_to_tailbase_relative_to_body_length = 'DistanceHindpawsToTailBaseMinusBodyLength'  # 3
    feat_dist_bw_front_paws = 'DistanceBetweenForepaws'  # 4
    feat_snout_speed = 'SnoutSpeed'  # 5
    feat_tail_base_speed = 'TailSpeed'  # 6
    feat_snout_tail_delta_angle = 'DeltaAngleTailToSnout'  # 7

    feat_dist_nose_obj1 = 'DistanceNosetipToObj1'
    feat_velocity_nose_obj1 = 'VelocityNosetipToObj1'
    feat_dist_nose_obj2 = 'DistanceNosetipToObj2'
    feat_velocity_nose_obj2 = 'VelocityNosetipToObj2'
    feat_dist_nose_obj3 = 'DistanceNosetipToObj3'
    feat_velocity_nose_obj3 = 'VelocityNosetipToObj3'
    feat_dist_nose_obj4 = 'DistanceNosetipToObj4'
    feat_velocity_nose_obj4 = 'VelocityNosetipToObj4'
    feat_dist_nose_obj5 = 'DistanceNosetipToObj5'
    feat_velocity_nose_obj5 = 'VelocityNosetipToObj5'
    feat_dist_nose_obj6 = 'DistanceNosetipToObj6'
    feat_velocity_nose_obj6 = 'VelocityNosetipToObj6'


    tau_of_3_intermediate_bodypart_avgForepaw = 'Tau3AvgForepaw'
    tau_of_3_intermediate_bodypart_avgHindpaw = 'Tau3AvgHindpaw'
    tau_of_3_feat_name_dist_forepawleft_nosetip = 'Tau3DistanceForepawLeftToNosetip'
    tau_of_3_feat_name_dist_forepawright_nosetip = 'Tau3DistanceForepawRightToNosetip'
    tau_of_3_feat_name_dist_forepawLeft_hindpawLeft = 'Tau3DistanceForepawLeftToHindpawLeft'
    tau_of_3_feat_name_dist_forepawRight_hindpawRight = 'Tau3DistanceForepawRightToHindpawRight'
    tau_of_3_feat_name_dist_AvgHindpaw_Nosetip = 'Tau3DistanceAvgHindpawToNosetip'
    tau_of_3_feat_name_dist_AvgForepaw_NoseTip = 'Tau3DistanceAvgForepawToNosetip'
    tau_of_3_feat_name_velocity_AvgForepaw = 'Tau3VelocityAvgForepaw'

    tau_of_3_feat_body_length = 'Tau3BodyLength'  # 1
    tau_of_3_intermediate_dist_avgForepaw_to_tailbase = 'Tau3DistanceAvgForepawToTailbase'
    tau_of_3_feat_dist_front_paws_to_tailbase_relative_to_body_length = 'Tau3DistanceForepawsToTailbaseMinusBodyLength'  # 2
    tau_of_3_intermediate_dist_avgHindpaw_to_tailbase = 'Tau3DistanceAvgHindpawToTailbase'
    tau_of_3_feat_dist_hind_paws_to_tailbase_relative_to_body_length = 'Tau3DistanceHindpawsToTailBaseMinusBodyLength'  # 3
    tau_of_3_feat_dist_bw_front_paws = 'Tau3DistanceBetweenForepaws'  # 4
    tau_of_3_feat_snout_speed = 'Tau3SnoutSpeed'  # 5
    tau_of_3_feat_tail_base_speed = 'Tau3TailSpeed'  # 6
    tau_of_3_feat_snout_tail_delta_angle = 'Tau3DeltaAngleTailToSnout'  # 7

    _all_engineered_features = (
        feat_name_dist_forepawleft_nosetip,
        feat_name_dist_forepawright_nosetip,
        feat_name_dist_forepawLeft_hindpawLeft,
        feat_name_dist_forepawRight_hindpawRight,
        feat_name_dist_AvgHindpaw_Nosetip,
        feat_name_dist_AvgForepaw_NoseTip,
        feat_name_velocity_AvgForepaw,
        feat_body_length,
        intermediate_dist_avgForepaw_to_tailbase,  # 1
        feat_dist_front_paws_to_tailbase_relative_to_body_length,  # 2
        intermediate_dist_avgHindpaw_to_tailbase,
        feat_dist_hind_paws_to_tailbase_relative_to_body_length,  # 3
        feat_dist_bw_front_paws,  # 4
        feat_snout_speed,  # 5
        feat_tail_base_speed,  # 6
        feat_snout_tail_delta_angle,  # 7
        feat_dist_nose_obj1,
        feat_velocity_nose_obj1,
        feat_dist_nose_obj2,
        feat_velocity_nose_obj2,
        feat_dist_nose_obj3,
        feat_velocity_nose_obj3,
        feat_dist_nose_obj4,
        feat_velocity_nose_obj4,
        feat_dist_nose_obj5,
        feat_velocity_nose_obj5,
        feat_dist_nose_obj6,
        feat_velocity_nose_obj6,

        tau_of_3_intermediate_bodypart_avgForepaw+'_x',
        tau_of_3_intermediate_bodypart_avgForepaw+'_y',
        tau_of_3_intermediate_bodypart_avgHindpaw+'_x',
        tau_of_3_intermediate_bodypart_avgHindpaw+'_y',
        tau_of_3_feat_name_dist_forepawleft_nosetip,
        tau_of_3_feat_name_dist_forepawright_nosetip,
        tau_of_3_feat_name_dist_forepawLeft_hindpawLeft,
        tau_of_3_feat_name_dist_forepawRight_hindpawRight,
        tau_of_3_feat_name_dist_AvgHindpaw_Nosetip,
        tau_of_3_feat_name_dist_AvgForepaw_NoseTip,
        tau_of_3_feat_name_velocity_AvgForepaw,

        tau_of_3_feat_body_length,
        tau_of_3_intermediate_dist_avgForepaw_to_tailbase,
        tau_of_3_feat_dist_front_paws_to_tailbase_relative_to_body_length,
        tau_of_3_intermediate_dist_avgHindpaw_to_tailbase,
        tau_of_3_feat_dist_hind_paws_to_tailbase_relative_to_body_length,
        tau_of_3_feat_dist_bw_front_paws,
        tau_of_3_feat_snout_speed,
        tau_of_3_feat_tail_base_speed,
        tau_of_3_feat_snout_tail_delta_angle,
    )

    def engineer_features(self, in_df: pd.DataFrame):
        # TODO: WIP
        """
        # Head dips
        1. d(forepaw left to nose)
        2. d(forepaw right to nose)
        # Rears
        3. d(forepaw left to hindpaw left)
        4. d(forepaw right to hindpaw right)
        5. d(nose to avg hindpaw)
        # Stretch attends
        6. d(avg hindpaw to nose) - same as #5
        7. d(avg forepaw to nose)
        8. v(avgForepaw)

        """
        # Arg Checking
        check_arg.ensure_type(in_df, pd.DataFrame)
        expected_cols = ['Object1_y', 'Object2_x']
        check_arg.ensure_columns_in_DataFrame(in_df, expected_cols)
        # Execute
        logger.debug(f'Engineering features for one data set...')
        df = in_df.sort_values('frame').copy()
        # Filter
        df, _ = feature_engineering.adaptively_filter_dlc_output(df)
        # Engineer features
        # 1
        df = feature_engineering.attach_feature_distance_between_2_bodyparts(df, config.get_part('FOREPAW_LEFT'), config.get_part('NOSETIP'), self.feat_name_dist_forepawleft_nosetip)
        # 2
        df = feature_engineering.attach_feature_distance_between_2_bodyparts(df, config.get_part('FOREPAW_RIGHT'), config.get_part('NOSETIP'), self.feat_name_dist_forepawright_nosetip)
        # 3
        df = feature_engineering.attach_feature_distance_between_2_bodyparts(df, config.get_part('FOREPAW_LEFT'), config.get_part('HINDPAW_LEFT'), self.feat_name_dist_forepawLeft_hindpawLeft)
        # 4
        df = feature_engineering.attach_feature_distance_between_2_bodyparts(df, config.get_part('FOREPAW_RIGHT'), config.get_part('HINDPAW_RIGHT'), self.feat_name_dist_forepawRight_hindpawRight)
        # 5, 6
        # df = feature_engineering.attach_average_forepaw_xy(df)  # BELOW SOLVES TODO: TODO: low: replace these two functions with the generalized xy averaging functions+output name?
        df = feature_engineering.attach_average_bodypart_xy(df, config.get_part('FOREPAW_LEFT'), config.get_part('FOREPAW_RIGHT'), output_bodypart=self.intermediate_bodypart_avgForepaw)

        # df = feature_engineering.attach_average_hindpaw_xy(df)  # BELO SOLVES TODO: low: replace these two functions with the generalized xy averaging functions+output name?
        df = feature_engineering.attach_average_bodypart_xy(df, config.get_part('HINDPAW_LEFT'), config.get_part('HINDPAW_RIGHT'), output_bodypart=self.intermediate_bodypart_avgHindpaw)

        df = feature_engineering.attach_feature_distance_between_2_bodyparts(df, self.intermediate_bodypart_avgHindpaw, config.get_part('NOSETIP'), self.feat_name_dist_AvgHindpaw_Nosetip)

        # 7
        # df = feature_engineering.attach_distance_between_2_feats(df, 'AvgForepaw', config.get_part('NOSETIP'), self.feat_name_dist_AvgForepaw_NoseTip)
        df = feature_engineering.attach_feature_distance_between_2_bodyparts(df, self.intermediate_bodypart_avgForepaw, config.get_part('NOSETIP'), self.feat_name_dist_AvgForepaw_NoseTip)

        # 8
        # df = feature_engineering.attach_velocity_of_feature(df, 'AvgForepaw', 1/config.VIDEO_FPS, self.feat_name_velocity_AvgForepaw)
        df = feature_engineering.attach_feature_velocity_of_bodypart(df, self.intermediate_bodypart_avgForepaw, action_duration=1 / config.VIDEO_FPS, output_feature_name=self.feat_name_velocity_AvgForepaw)

        #####################################################
        # 1 dist snout to tail
        df = feature_engineering.attach_feature_distance_between_2_bodyparts(df, config.get_part('TAILBASE'), config.get_part('NOSETIP'), output_feature_name=self.feat_body_length)

        # 2: Dist FrontPaws to tail relative to body length
        ## 1/3: Get AvgForepaw location
        df = feature_engineering.attach_average_bodypart_xy(df, config.get_part('FOREPAW_LEFT'), config.get_part('FOREPAW_RIGHT'), output_bodypart=self.intermediate_bodypart_avgForepaw)
        ## 2/3: Get dist from forepaw to tailbase
        df = feature_engineering.attach_feature_distance_between_2_bodyparts(df, self.intermediate_bodypart_avgForepaw, config.get_part('TAILBASE'), self.intermediate_dist_avgForepaw_to_tailbase)
        ## 3/3: Get body-length relative distance
        df[self.feat_dist_front_paws_to_tailbase_relative_to_body_length] = df[self.feat_body_length] - df[self.intermediate_dist_avgForepaw_to_tailbase]

        # 3 Dist back paws to base of tail relative to body length
        ## 1/3: Get AvgHindpaw location
        df = feature_engineering.attach_average_bodypart_xy(df, config.get_part('HINDPAW_LEFT'), config.get_part('HINDPAW_RIGHT'), output_bodypart=self.intermediate_bodypart_avgHindpaw)
        ## 2/3: Get dist from hindpaw to tailbase
        df = feature_engineering.attach_feature_distance_between_2_bodyparts(df, self.intermediate_bodypart_avgHindpaw, config.get_part('TAILBASE'), output_feature_name=self.intermediate_dist_avgHindpaw_to_tailbase)
        ## 3/3: Get body-length relative distance
        df[self.feat_dist_hind_paws_to_tailbase_relative_to_body_length] = df[self.feat_body_length] - df[self.intermediate_dist_avgHindpaw_to_tailbase]

        # 4: distance between 2 front paws
        df = feature_engineering.attach_feature_distance_between_2_bodyparts(df, config.get_part('FOREPAW_LEFT'), config.get_part('FOREPAW_RIGHT'), self.feat_dist_bw_front_paws)

        # 5: snout speed
        df = feature_engineering.attach_feature_velocity_of_bodypart(df, config.get_part('NOSETIP'), action_duration=1 / self.video_fps, output_feature_name=self.feat_snout_speed)

        # 6 tail speed
        df = feature_engineering.attach_feature_velocity_of_bodypart(df, config.get_part('TAILBASE'), action_duration=1 / self.video_fps, output_feature_name=self.feat_tail_base_speed)

        # 7: snout to base of tail change in angle
        df = feature_engineering.attach_angle_between_bodyparts(df, config.get_part('NOSETIP'), config.get_part('TAILBASE'), self.feat_snout_tail_delta_angle)

        logger.warn(f'COLS: {df.columns}')

        df = feature_engineering.attach_feature_distance_between_2_bodyparts(df, config.get_part('NOSETIP'), 'Object1', output_feature_name=self.feat_dist_nose_obj1)
        df = feature_engineering.attach_feature_distance_between_2_bodyparts(df, config.get_part('NOSETIP'), 'Object2', output_feature_name=self.feat_dist_nose_obj2)
        df = feature_engineering.attach_feature_distance_between_2_bodyparts(df, config.get_part('NOSETIP'), 'Object3', output_feature_name=self.feat_dist_nose_obj3)
        df = feature_engineering.attach_feature_distance_between_2_bodyparts(df, config.get_part('NOSETIP'), 'Object4', output_feature_name=self.feat_dist_nose_obj4)
        df = feature_engineering.attach_feature_distance_between_2_bodyparts(df, config.get_part('NOSETIP'), 'Object5', output_feature_name=self.feat_dist_nose_obj5)
        df = feature_engineering.attach_feature_distance_between_2_bodyparts(df, config.get_part('NOSETIP'), 'Object6', output_feature_name=self.feat_dist_nose_obj6)
        # DistanceNosetipToObj1_x
        df = feature_engineering.attach_delta_of_single_column(df, self.feat_dist_nose_obj1, action_duration=1/self.video_fps, output_feature_name=self.feat_velocity_nose_obj1)
        df = feature_engineering.attach_delta_of_single_column(df, self.feat_dist_nose_obj2, action_duration=1/self.video_fps, output_feature_name=self.feat_velocity_nose_obj2)
        df = feature_engineering.attach_delta_of_single_column(df, self.feat_dist_nose_obj3, action_duration=1/self.video_fps, output_feature_name=self.feat_velocity_nose_obj3)
        df = feature_engineering.attach_delta_of_single_column(df, self.feat_dist_nose_obj4, action_duration=1/self.video_fps, output_feature_name=self.feat_velocity_nose_obj4)
        df = feature_engineering.attach_delta_of_single_column(df, self.feat_dist_nose_obj5, action_duration=1/self.video_fps, output_feature_name=self.feat_velocity_nose_obj5)
        df = feature_engineering.attach_delta_of_single_column(df, self.feat_dist_nose_obj6, action_duration=1/self.video_fps, output_feature_name=self.feat_velocity_nose_obj6)

        tau = 3
        df = feature_engineering.attach_time_shifted_data(df, self.intermediate_bodypart_avgForepaw, tau=tau, output_feature_name=self.tau_of_3_intermediate_bodypart_avgForepaw)
        df = feature_engineering.attach_time_shifted_data(df, self.intermediate_bodypart_avgHindpaw, tau=tau, output_feature_name=self.tau_of_3_intermediate_bodypart_avgHindpaw)
        df = feature_engineering.attach_time_shifted_data(df, self.feat_name_dist_forepawleft_nosetip, tau=tau, output_feature_name=self.tau_of_3_feat_name_dist_forepawleft_nosetip)
        df = feature_engineering.attach_time_shifted_data(df, self.feat_name_dist_forepawright_nosetip, tau=tau, output_feature_name=self.tau_of_3_feat_name_dist_forepawright_nosetip)
        df = feature_engineering.attach_time_shifted_data(df, self.feat_name_dist_forepawLeft_hindpawLeft, tau=tau, output_feature_name=self.tau_of_3_feat_name_dist_forepawLeft_hindpawLeft)
        df = feature_engineering.attach_time_shifted_data(df, self.feat_name_dist_forepawRight_hindpawRight, tau=tau, output_feature_name=self.tau_of_3_feat_name_dist_forepawRight_hindpawRight)
        df = feature_engineering.attach_time_shifted_data(df, self.feat_name_dist_AvgHindpaw_Nosetip, tau=tau, output_feature_name=self.tau_of_3_feat_name_dist_AvgHindpaw_Nosetip)
        df = feature_engineering.attach_time_shifted_data(df, self.feat_name_dist_AvgHindpaw_Nosetip, tau=tau, output_feature_name=self.tau_of_3_feat_name_dist_AvgForepaw_NoseTip)
        df = feature_engineering.attach_time_shifted_data(df, self.feat_name_velocity_AvgForepaw, tau=tau, output_feature_name=self.tau_of_3_feat_name_velocity_AvgForepaw)

        df = feature_engineering.attach_time_shifted_data(df, self.feat_body_length, tau=tau, output_feature_name=self.tau_of_3_feat_body_length)
        df = feature_engineering.attach_time_shifted_data(df, self.intermediate_dist_avgForepaw_to_tailbase, tau=tau, output_feature_name=self.tau_of_3_intermediate_dist_avgForepaw_to_tailbase)
        df = feature_engineering.attach_time_shifted_data(df, self.feat_dist_front_paws_to_tailbase_relative_to_body_length, tau=tau, output_feature_name=self.tau_of_3_feat_dist_front_paws_to_tailbase_relative_to_body_length)
        df = feature_engineering.attach_time_shifted_data(df, self.intermediate_dist_avgHindpaw_to_tailbase, tau=tau, output_feature_name=self.tau_of_3_intermediate_dist_avgHindpaw_to_tailbase)
        df = feature_engineering.attach_time_shifted_data(df, self.feat_dist_hind_paws_to_tailbase_relative_to_body_length, tau=tau, output_feature_name=self.tau_of_3_feat_dist_hind_paws_to_tailbase_relative_to_body_length)
        df = feature_engineering.attach_time_shifted_data(df, self.feat_dist_bw_front_paws, tau=tau, output_feature_name=self.tau_of_3_feat_dist_bw_front_paws)
        df = feature_engineering.attach_time_shifted_data(df, self.feat_snout_speed, tau=tau, output_feature_name=self.tau_of_3_feat_snout_speed)
        df = feature_engineering.attach_time_shifted_data(df, self.feat_tail_base_speed, tau=tau, output_feature_name=self.tau_of_3_feat_tail_base_speed)
        df = feature_engineering.attach_time_shifted_data(df, self.feat_snout_tail_delta_angle, tau=tau, output_feature_name=self.tau_of_3_feat_snout_tail_delta_angle)

        # BINNING #
        map_feature_to_integrate_method = {
            self.feat_name_dist_forepawleft_nosetip: 'avg',
            self.feat_name_dist_forepawright_nosetip: 'avg',
            self.feat_name_dist_forepawLeft_hindpawLeft: 'avg',
            self.feat_name_dist_forepawRight_hindpawRight: 'avg',
            self.feat_name_dist_AvgHindpaw_Nosetip: 'avg',
            self.feat_name_dist_AvgForepaw_NoseTip: 'avg',
            self.feat_name_velocity_AvgForepaw: 'sum',

            self.feat_body_length: 'avg',
            self.feat_dist_front_paws_to_tailbase_relative_to_body_length: 'avg',
            self.feat_dist_hind_paws_to_tailbase_relative_to_body_length: 'avg',
            self.feat_dist_bw_front_paws: 'avg',
            self.feat_snout_speed: 'sum',
            self.feat_tail_base_speed: 'sum',
            self.feat_snout_tail_delta_angle: 'sum',

            self.feat_dist_nose_obj1: 'avg',
            self.feat_velocity_nose_obj1: 'avg',
            self.feat_dist_nose_obj2: 'avg',
            self.feat_velocity_nose_obj2: 'avg',
            self.feat_dist_nose_obj3: 'avg',
            self.feat_velocity_nose_obj3: 'avg',
            self.feat_dist_nose_obj4: 'avg',
            self.feat_velocity_nose_obj4: 'avg',
            self.feat_dist_nose_obj5: 'avg',
            self.feat_velocity_nose_obj5: 'avg',
            self.feat_dist_nose_obj6: 'avg',
            self.feat_velocity_nose_obj6: 'avg',

            self.tau_of_3_intermediate_bodypart_avgForepaw+'_x': 'avg',
            self.tau_of_3_intermediate_bodypart_avgForepaw+'_y': 'avg',
            self.tau_of_3_intermediate_bodypart_avgHindpaw+'_x': 'avg',
            self.tau_of_3_intermediate_bodypart_avgHindpaw+'_y': 'avg',
            self.tau_of_3_feat_name_dist_forepawleft_nosetip: 'avg',
            self.tau_of_3_feat_name_dist_forepawright_nosetip: 'avg',
            self.tau_of_3_feat_name_dist_forepawLeft_hindpawLeft: 'avg',
            self.tau_of_3_feat_name_dist_forepawRight_hindpawRight: 'avg',
            self.tau_of_3_feat_name_dist_AvgHindpaw_Nosetip: 'avg',
            self.tau_of_3_feat_name_dist_AvgForepaw_NoseTip: 'avg',
            self.tau_of_3_feat_name_velocity_AvgForepaw: 'avg',

            self.tau_of_3_feat_body_length: 'avg',
            self.tau_of_3_intermediate_dist_avgForepaw_to_tailbase: 'avg',
            self.tau_of_3_feat_dist_front_paws_to_tailbase_relative_to_body_length: 'avg',
            self.tau_of_3_intermediate_dist_avgHindpaw_to_tailbase: 'avg',
            self.tau_of_3_feat_dist_hind_paws_to_tailbase_relative_to_body_length: 'avg',
            self.tau_of_3_feat_dist_bw_front_paws: 'avg',
            self.tau_of_3_feat_snout_speed: 'avg',
            self.tau_of_3_feat_tail_base_speed: 'avg',
            self.tau_of_3_feat_snout_tail_delta_angle: 'avg',
        }

        logger.debug(f'{get_current_function()}(): # of rows in DataFrame before binning = {len(df)}')
        df = feature_engineering.integrate_df_feature_into_bins(df, map_feature_to_integrate_method, self.average_over_n_frames)
        logger.debug(f'{get_current_function()}(): # of rows in DataFrame after binning = {len(df)}')

        # # Debug effort/check: ensure columns don't get dropped by accident
        # for col in in_df.columns:
        #     if col not in list(df.columns):
        #         err_missing_col = f'Missing col should not have been lost in feature engineering but was. ' \
        #                           f'Column = {col}. (df={df.head().to_string()})'  # TODO: low: improve err message
        #         logger.error(err_missing_col)
        #         raise KeyError(err_missing_col)

        logger.debug(f'Done engineering features.')
        return df


### Accessory functions ###

def generate_pipeline_filename(name: str):
    """
    Generates a pipeline file name given its name.

    This is an effort to standardize naming for saving pipelines.
    """
    file_name = f'{name}.pipeline'
    return file_name


def generate_pipeline_filename_from_pipeline(pipeline_obj: BasePipeline) -> str:
    return generate_pipeline_filename(pipeline_obj.name)


def generate_standardized_Howland_pipeline(name: str) -> BasePipeline:
    # TODO: med: remove this function after debugging
    """
    Tool to quickly create a Pipeline with data
    :param name:
    :return:
    """
    p = PipelineHowland(name)
    p = p.add_train_data_source(config.DEFAULT_TRAIN_DATA_DIR)
    p = p.add_predict_data_source(config.DEFAULT_TEST_DATA_DIR)

    return p

