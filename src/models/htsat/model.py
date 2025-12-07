import torch
import lightning as L
import torch.optim as optim
import bisect
import os
from loguru import logger
import torch
from torch.utils.data import DataLoader

from external.hts_audio_transformer.utils import get_loss_func
from src.config import CLASSES, DETECTION_TRAIN_PATH, GROUND_TRUTH_EXTRACTED_AUDIO_PATH
from src.models.htsat.data import process_data_for_train, process_train_data_for_classification, HTSATDataset, format_dataset
from src.models.htsat.constants import NON_EVENT_LABEL, CLASS_LABELS, DEVICE, FRAME_LOSS_BASIS, CLIP_LOSS_BASIS


class HTSATModel(L.LightningModule):
    def __init__(self, model, config, loss_basis):
        super().__init__()
        self.model = model
        self.config = config
        # self.dataset = dataset
        self.loss_func = get_loss_func(config.loss_type)
        self.test_step_outputs = []
        self.loss_basis = loss_basis
    
    def forward(self, x, mix_lambda = None):
        output_dict = self.model(x)
        return output_dict["clipwise_output"] , output_dict["framewise_output"]
    
    def training_step(self, batch, batch_idx):
        self.device_type = next(self.parameters()).device
        mix_lambda = None

        batch_waveform = batch['waveform']
        pred, pred_map = self(batch_waveform, mix_lambda)

        if self.loss_basis == FRAME_LOSS_BASIS:
            loss = self.calculate_framewise_loss(batch, pred_map)
        elif self.loss_basis == CLIP_LOSS_BASIS:
            loss = self.calculate_clipwise_loss(batch, pred)
        else:
            exit(1)

        self.log("loss", loss, on_epoch= True, prog_bar=True, batch_size=self.config.batch_size)

        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr = self.config.learning_rate, 
            betas = (0.9, 0.999), eps = 1e-08, weight_decay = 0.05, 
        )
        # Change: SWA, deprecated
        # optimizer = SWA(optimizer, swa_start=10, swa_freq=5)
        def lr_foo(epoch):       
            if epoch < 3:
                # warm up lr
                lr_scale = self.config.lr_rate[epoch]
            else:
                # warmup schedule
                lr_pos = int(-1 - bisect.bisect_left(self.config.lr_scheduler_epoch, epoch))
                if lr_pos < -3:
                    lr_scale = max(self.config.lr_rate[0] * (0.98 ** epoch), 0.03 )
                else:
                    lr_scale = self.config.lr_rate[lr_pos]
            return lr_scale
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lr_foo
        )
        
        return [optimizer], [scheduler]
    
    def validation_step(self, batch, batch_idx):
        pred, _ = self(batch["waveform"])
        return [pred.detach(), batch["target"].detach()]
    
    def time_shifting(self, x, shift_len):
        shift_len = int(shift_len)
        new_sample = torch.cat([x[:, shift_len:], x[:, :shift_len]], axis = 1)
        return new_sample 
    
    def test_step(self, batch, batch_idx):
        pred, pred_map = self(batch["waveform"])    # pred: (B, C) | pred_map: (B, window_size, C)
        target = batch['target']                    # target: (B, C)
        onset = batch['onset']
        offset = batch['offset']
        loss = self.loss_func(pred, target)

        output = {
            'file': batch['file'],
            'file_path': batch['file_path'],
            'event_label': batch['event_label'],
            'pred': pred.detach().cpu(), 
            'pred_map': pred_map.detach().cpu(),
            'target': target.detach().cpu(),
            'onset': onset.detach().cpu(),
            'offset': offset.detach().cpu(),
        }
        self.test_step_outputs.append(output)

        return

    def on_test_epoch_end(self):
        test_step_outputs = self.test_step_outputs

    def calculate_clipwise_loss(self, batch, pred):
        target = batch['target']
        clip_loss = self.loss_func(pred, target)
        return clip_loss
    
    def calculate_framewise_loss(self, batch, pred_map):
        self.device_type = next(self.parameters()).device

        batch_size, num_frames, num_classes = pred_map.shape

        # Frame-Wise Loss Calculation
        # Construct the Frame-wise Target Map
        batch_target = batch['target'].to(self.device_type).float() # (shape: 10, 6)
        batch_onset = batch['onset'].to(self.device_type).float()  # (shape: 10)
        batch_offset = batch['offset'].to(self.device_type).float()   # (shape: 10)

        target_map = torch.zeros(batch_size, num_frames, num_classes, device=self.device_type)
        target_map[:, :, self.config.classes.str2int(NON_EVENT_LABEL)] = 1

        frame_onset = batch_onset / self.config.clip_duration * num_frames   # (shape: 10)
        frame_offset = batch_offset / self.config.clip_duration * num_frames   # (shape: 10)

        frame_onset = frame_onset.round().long()
        frame_offset = frame_offset.round().long()

        onset_broadcast = frame_onset.unsqueeze(1)    # Shape (10, 1)
        offset_broadcast = frame_offset.unsqueeze(1)  # Shape (10, 1)

        frame_indices = torch.arange(num_frames, device=self.device_type)
        mask = (frame_indices >= onset_broadcast) & (frame_indices < offset_broadcast)
        mask_expanded = mask.unsqueeze(-1).expand_as(target_map) # (10, 1024, 6)target_expanded = batch_target.unsqueeze(1) # (10, 1, 6)

        target_expanded = batch_target.unsqueeze(1) # (10, 1, 6)
        target_map = torch.where(mask_expanded, target_expanded, target_map)

        loss = self.loss_func(pred_map, target_map)

        return loss
    
def init_model(config, checkpoint_path, loss_basis, test_only = False):
    logger.info("Intializing HTS-AT Model")
    from external.hts_audio_transformer.model.htsat import HTSAT_Swin_Transformer

    sed_model = HTSAT_Swin_Transformer(
        spec_size=config.htsat_spec_size,
        num_classes=config.classes_num,
        config = config
    )

    sed_model.to(DEVICE)

    if os.path.exists(checkpoint_path):
        logger.info(f"HTS-AT Model checkpoint exists at {checkpoint_path}, Loading weights...")
        state_dict = torch.load(checkpoint_path, map_location=DEVICE)
        sed_model.load_state_dict(state_dict)

    if test_only:
        sed_model.eval()

    model = HTSATModel(sed_model, config, loss_basis)

    return model

def get_config():
    from external.hts_audio_transformer import config

    ## Add / Modify Configurations
    config.debug = True
    config.max_epoch = 1
    config.classes_num = len(CLASSES) + 1
    config.sample_rate = 44100
    config.batch_size = 10

    config.clip_duration = 20.0
    config.classes = CLASS_LABELS
    config.clip_samples = config.sample_rate * config.clip_duration
    config.htsat_spec_size = 512

    return config


def run_htsat_train(checkpoint_path, loss_basis):

    config = get_config()
    model = init_model(config, checkpoint_path, loss_basis)
    
    train_dataset = process_data_for_train()

    formatted_train_dataset = format_dataset(train_dataset, CLASS_LABELS)
    htsat_train_dataset = HTSATDataset(formatted_train_dataset, config)

    train_loader = DataLoader(
        dataset = htsat_train_dataset,
        num_workers = config.num_workers,
        batch_size = config.batch_size,
        shuffle = False,
    )

    minimal_trainer = L.Trainer(
        max_epochs=config.max_epoch,
        default_root_dir="./lightning_checkpoints",
    )

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    minimal_trainer.fit(model, train_dataloaders=train_loader)

    trained_model = model.model
    torch.save(trained_model.state_dict(), checkpoint_path)

    return 


def run_htsat_train_classification(checkpoint_path, loss_basis):

    from src.utils.generate_ground_truth import generate_gt_events_dict
    from src.utils.cut_audio import cut_events_from_audio

    config = get_config()
    config.batch_size = 1
    model = init_model(config, checkpoint_path, loss_basis)

    gt_train_events_list = generate_gt_events_dict('data/processed/yamnet/spectrograms_train_list.pkl')

    updated_gt_train_events_list = cut_events_from_audio(
        GROUND_TRUTH_EXTRACTED_AUDIO_PATH, gt_train_events_list, DETECTION_TRAIN_PATH
    )
    
    train_dataset = process_train_data_for_classification(updated_gt_train_events_list)

    formatted_train_dataset = format_dataset(train_dataset, CLASS_LABELS)
    htsat_train_dataset = HTSATDataset(formatted_train_dataset, config)

    train_loader = DataLoader(
        dataset = htsat_train_dataset,
        num_workers = config.num_workers,
        batch_size = config.batch_size,
        shuffle = False,
    )

    minimal_trainer = L.Trainer(
        max_epochs=config.max_epoch,
        default_root_dir="./lightning_checkpoints",
    )

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    minimal_trainer.fit(model, train_dataloaders=train_loader)

    trained_model = model.model
    torch.save(trained_model.state_dict(), checkpoint_path)

    return 
