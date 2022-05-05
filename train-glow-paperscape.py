import os

# Trainer: Where the ✨️ happens.
# TrainingArgs: Defines the set of arguments of the Trainer.
from trainer import Trainer, TrainerArgs

# GlowTTSConfig: all model related values for training, validating and testing.
from TTS.tts.configs.glow_tts_config import GlowTTSConfig

# BaseDatasetConfig: defines name, formatter and path of the dataset.
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.glow_tts import GlowTTS
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
from TTS.tts.configs.shared_configs import CharactersConfig

# we use the same path as this script as our training folder.
output_path = '/storage/output'


def formatter(root_path, meta_file, **kwargs):  # pylint: disable=unused-argument
    txt_file = os.path.join(root_path, meta_file)
    items = []
    speaker_name = "maledataset1"
    with open(txt_file, "r", encoding="utf-8") as ttf:
        for line in ttf:
            cols = line.split("|")
            wav_file = os.path.join(root_path, "wavs", cols[0])
            text = cols[1]
            items.append({"text": text, "audio_file": wav_file, "speaker_name":speaker_name})
    return items

# DEFINE DATASET CONFIG
# Set LJSpeech as our target dataset and define its path.
# You can also use a simple Dict to define the dataset and pass it to your custom formatter.
dataset_config = BaseDatasetConfig(
    name="commonvoice-be", meta_file_train="metadata.csv", path='/storage/be'
)
characters=CharactersConfig(
    characters_class="TTS.tts.utils.text.characters.Graphemes",
    pad="_",
    eos="~",
    bos="^",
    blank="@",
    characters="\u0430\u0431\u0432\u0433\u0434\u0435\u0451\u0436\u0437\u0456\u0439\u043a\u043b\u043c\u043d\u043e\u043f\u0440\u0441\u0442\u0443\u045e\u0444\u0445\u0446\u0447\u0448\u044b\u044c\u044d\u044e\u044f'",
    punctuations="!,.? —",
)
# INITIALIZE THE TRAINING CONFIGURATION
# Configure the model. Every config class inherits the BaseTTSConfig.
config = GlowTTSConfig(
    batch_size=64,
    eval_batch_size=32,
    mixed_precision=False,
    use_grad_scaler=False,
    num_loader_workers=8,
    num_eval_loader_workers=8,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=1000,
    text_cleaner="belarusian_cleaners",
    use_phonemes=False,
    print_step=50,
    print_eval=True,
    use_noise_augment=True,
    output_path=output_path,
    datasets=[dataset_config],
    characters=characters,
    add_blank=True,
    enable_eos_bos_chars=True,
    save_step=10000,
    save_n_checkpoints=2,
    save_all_best=False,
    save_best_after=5000,
    test_sentences=[
        "Тэставы сказ.",
        "У рудога вераб’я ў сховішчы пад фатэлем ляжаць нейкія гаючыя зёлкі",
        "Я жорстка заб’ю проста ў сэрца гэты расквечаны профіль, што ходзіць ля маёй хаты"
    ],
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
        "do_sound_norm": False,
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
        "spec_gain": 1,
        "do_amp_to_db_linear": True,
        "do_amp_to_db_mel": True,
        "pitch_fmax": 640.0,
        "pitch_fmin": 0.0,
        "signal_norm": True,
        "min_level_db": -100,
        "symmetric_norm": True,
        "max_norm": 4.0,
        "clip_norm": True,
        "stats_path": 'scale_stats.npy'
    }
)
# config_characters = BaseCharacters(**config.characters)
# INITIALIZE THE AUDIO PROCESSOR
# Audio processor is used for feature extraction and audio I/O.
# It mainly serves to the dataloader and the training loggers.
ap = AudioProcessor.init_from_config(config)

# INITIALIZE THE TOKENIZER
# Tokenizer is used to convert text to sequences of token IDs.
# If characters are not defined in the config, default characters are passed to the config
tokenizer, config = TTSTokenizer.init_from_config(config)

# LOAD DATA SAMPLES
# Each sample is a list of ```[text, audio_file_path, speaker_name]```
# You can define your custom sample loader returning the list of samples.
# Or define your custom formatter and pass it to the `load_tts_samples`.
# Check `TTS.tts.datasets.load_tts_samples` for more details.
train_samples, eval_samples = load_tts_samples(dataset_config, eval_split=True, formatter=formatter)

# INITIALIZE THE MODEL
# Models take a config object and a speaker manager as input
# Config defines the details of the model like the number of layers, the size of the embedding, etc.
# Speaker manager is used by multi-speaker models.
model = GlowTTS(config, ap, tokenizer, speaker_manager=None)

# INITIALIZE THE TRAINER
# Trainer provides a generic API to train all the 🐸TTS models with all its perks like mixed-precision training,
# distributed training, etc.
trainer = Trainer(
    TrainerArgs(), config, output_path, model=model, train_samples=train_samples, eval_samples=eval_samples
)

# AND... 3,2,1... 🚀
trainer.fit()