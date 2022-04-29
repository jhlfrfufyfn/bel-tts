import os

from trainer import Trainer, TrainerArgs

from TTS.utils.audio import AudioProcessor
from TTS.vocoder.configs import HifiganConfig
from TTS.vocoder.datasets.preprocess import load_wav_data
from TTS.vocoder.models.gan import GAN

output_path = '/storage/output'

config = HifiganConfig(
    batch_size=64,
    eval_batch_size=32,
    num_loader_workers=8,
    num_eval_loader_workers=8,
    run_eval=True,
    test_delay_epochs=5,
    epochs=1000,
    seq_len=8192,
    pad_short=2000,
    use_noise_augment=False,
    eval_split_size=10,
    print_step=25,
    print_eval=True,
    mixed_precision=True,
    save_n_checkpoints=1,
    lr_gen=1e-4,
    lr_disc=1e-4,
    data_path="/storage/be",
    output_path=output_path,
    audio={
        "fft_size": 1024,
        "win_length": 1024,
        "hop_length": 256,
        "frame_shift_ms": None,
        "frame_length_ms": None,
        "stft_pad_mode": "reflect",
        "sample_rate": 16000,
        "resample": False,
        "preemphasis": 0.0,
        "ref_level_db": 20,
        "do_sound_norm": True,
        "log_func": "np.log10",
        "do_trim_silence": True,
        "trim_db": 45,
        "do_rms_norm": False,
        "db_level": None,
        "power": 1.5,
        "griffin_lim_iters": 60,
        "num_mels": 80,
        "mel_fmin": 50,
        "mel_fmax": 8000,
        "spec_gain": 20,
        "do_amp_to_db_linear": True,
        "do_amp_to_db_mel": True,
        "pitch_fmax": 640.0,
        "pitch_fmin": 0.0,
        "signal_norm": True,
        "min_level_db": -100,
        "symmetric_norm": True,
        "max_norm": 4.0,
        "clip_norm": True,
        "stats_path": None
    },
    l1_spec_loss_params={
        "use_mel": True,
        "sample_rate": 16000,
        "n_fft": 1024,
        "hop_length": 256,
        "win_length": 1024,
        "n_mels": 80,
        "mel_fmin": 50.0,
        "mel_fmax": 8000,
    }
)

# init audio processor
ap = AudioProcessor.init_from_config(config)

# load training samples
eval_samples, train_samples = load_wav_data(config.data_path, config.eval_split_size)

# init model
model = GAN(config, ap)

# init the trainer and 🚀
trainer = Trainer(
    TrainerArgs(), config, output_path, model=model, train_samples=train_samples, eval_samples=eval_samples
)
trainer.fit()