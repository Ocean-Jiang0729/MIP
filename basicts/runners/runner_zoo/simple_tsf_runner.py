import torch
import csv
import numpy as np
import os
from ..base_tsf_runner import BaseTimeSeriesForecastingRunner


class SimpleTimeSeriesForecastingRunner(BaseTimeSeriesForecastingRunner):
    """Simple Runner: select forward features and target features. This runner can cover most cases."""

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.forward_features = cfg["MODEL"].get("FORWARD_FEATURES", None)
        self.target_features = cfg["MODEL"].get("TARGET_FEATURES", None)

    def select_input_features(self, data: torch.Tensor) -> torch.Tensor:
        """Select input features.

        Args:
            data (torch.Tensor): input history data, shape [B, L, N, C]

        Returns:
            torch.Tensor: reshaped data
        """

        # select feature using self.forward_features
        if self.forward_features is not None:
            data = data[:, :, :, self.forward_features]
        return data

    def select_target_features(self, data: torch.Tensor) -> torch.Tensor:
        """Select target feature.

        Args:
            data (torch.Tensor): prediction of the model with arbitrary shape.

        Returns:
            torch.Tensor: reshaped data with shape [B, L, N, C]
        """

        # select feature using self.target_features
        data = data[:, :, :, self.target_features]
        return data

    def forward(self, data: tuple, epoch: int = None, iter_num: int = None, train: bool = True, **kwargs) -> tuple:
        """Feed forward process for train, val, and test. Note that the outputs are NOT re-scaled.

        Args:
            data (tuple): data (future data, history ata).
            epoch (int, optional): epoch number. Defaults to None.
            iter_num (int, optional): iteration number. Defaults to None.
            train (bool, optional): if in the training process. Defaults to True.

        Returns:
            dict: keys that must be included: inputs, prediction, target
        """

        # preprocess
        future_data, history_data = data
        history_data = self.to_running_device(history_data)      # B, L, N, C
        future_data = self.to_running_device(future_data)       # B, L, N, C
        batch_size, length, num_nodes, _ = future_data.shape
        #print("history_data{}, future_data{}".format(history_data.shape,future_data.shape))
        history_data = self.select_input_features(history_data)
        if train:
            future_data_4_dec = self.select_input_features(future_data)
        else:
            future_data_4_dec = self.select_input_features(future_data)
            # only use the temporal features
            future_data_4_dec[..., 0] = torch.empty_like(future_data_4_dec[..., 0])

        # model forward
        model_return = self.model(history_data=history_data, future_data=future_data_4_dec, batch_seen=iter_num, epoch=epoch, train=train)

        # parse model return
        if isinstance(model_return, torch.Tensor): model_return = {"prediction": model_return}
        if "inputs" not in model_return: model_return["inputs"] = self.select_target_features(history_data)
        if "target" not in model_return: model_return["target"] = self.select_target_features(future_data)
        assert list(model_return["prediction"].shape)[:3] == [batch_size, length, num_nodes], \
            "error shape of the output, edit the forward function to reshape it to [B, L, N, C]"
        # node_number = 33
        # save_dir = "/home/jhy/BasicTS-master/BasicTS-master/save_csv"
        # target_filename = os.path.join(save_dir, 'target2.txt')
        # prediction_filename = os.path.join(save_dir, 'prediction.txt')
        # # score_filename = os.path.join(save_dir, 'score.txt')
        # target_array = model_return["target"][:, :, node_number, 0].cpu().detach().numpy()#.reshape(1,12)
        # prediction_array = model_return["prediction"][:, :, node_number, 0].cpu().detach().numpy()#.reshape(1,12)
        # invariant_score = self.model.gate.score1[:,:,node_number,:].cpu().detach().numpy().reshape(-1,12*30)
        # #print("self.model.gate.score1[:,:,node_number,:]", self.model.gate.score1[:,:,node_number,:].shape)
        # #if prediction_array.shape[0]!=invariant_score.shape[0]:
        # #    print("prediction_array.shape[0]:{},invariant_score.shape[0]:{}".format(prediction_array.shape[0],invariant_score.shape[0]))
        # #print(invariant_score.shape)
        # # Open the CSV file in append mode
        # with open(target_filename, 'a') as file:
        #     tensor_string = '\n'.join(' '.join(map(str, row)) for row in target_array) + '\n'
        #     file.write(tensor_string)
            
        # # # Open the CSV file in append mode
        # with open(prediction_filename, 'a') as file:
        #     tensor_string = '\n'.join(' '.join(map(str, row)) for row in prediction_array) + '\n'
        #     file.write(tensor_string)

        # with open(score_filename, 'a') as file:
        #     tensor_string = '\n'.join(' '.join(map(str, row)) for row in invariant_score) + '\n'
        #     file.write(tensor_string)
        

        #print("target", model_return["target"][0, :, 33, 0])
        #print("prediction", model_return["prediction"][0, :, 33, 0])
        return model_return
