import argparse
import logging
from typing import Callable
from typing import Collection
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
import os
from pathlib import Path
from typing import Tuple
from typing import Union
import yaml
import numpy as np
import torch
from typeguard import check_argument_types
from typeguard import check_return_type

from funasr_local.datasets.collate_fn import CommonCollateFn
from funasr_local.datasets.preprocessor import CommonPreprocessor
from funasr_local.models.ctc import CTC
from funasr_local.models.decoder.abs_decoder import AbsDecoder
from funasr_local.models.decoder.rnn_decoder import RNNDecoder
from funasr_local.models.decoder.transformer_decoder import (
    DynamicConvolution2DTransformerDecoder,  # noqa: H301
)
from funasr_local.models.decoder.transformer_decoder import DynamicConvolutionTransformerDecoder
from funasr_local.models.decoder.transformer_decoder import (
    LightweightConvolution2DTransformerDecoder,  # noqa: H301
)
from funasr_local.models.decoder.transformer_decoder import (
    LightweightConvolutionTransformerDecoder,  # noqa: H301
)
from funasr_local.models.decoder.transformer_decoder import TransformerDecoder
from funasr_local.models.encoder.abs_encoder import AbsEncoder
from funasr_local.models.encoder.conformer_encoder import ConformerEncoder
from funasr_local.models.encoder.data2vec_encoder import Data2VecEncoder
from funasr_local.models.encoder.rnn_encoder import RNNEncoder
from funasr_local.models.encoder.transformer_encoder import TransformerEncoder
from funasr_local.models.frontend.abs_frontend import AbsFrontend
from funasr_local.models.frontend.default import DefaultFrontend
from funasr_local.models.frontend.fused import FusedFrontends
from funasr_local.models.frontend.wav_frontend import WavFrontend, WavFrontendOnline
from funasr_local.models.frontend.s3prl import S3prlFrontend
from funasr_local.models.frontend.windowing import SlidingWindow
from funasr_local.models.postencoder.abs_postencoder import AbsPostEncoder
from funasr_local.models.postencoder.hugging_face_transformers_postencoder import (
    HuggingFaceTransformersPostEncoder,  # noqa: H301
)
from funasr_local.models.preencoder.abs_preencoder import AbsPreEncoder
from funasr_local.models.preencoder.linear import LinearProjection
from funasr_local.models.preencoder.sinc import LightweightSincConvs
from funasr_local.models.specaug.abs_specaug import AbsSpecAug
from funasr_local.models.specaug.specaug import SpecAug
from funasr_local.layers.abs_normalize import AbsNormalize
from funasr_local.layers.global_mvn import GlobalMVN
from funasr_local.layers.utterance_mvn import UtteranceMVN
from funasr_local.tasks.abs_task import AbsTask
from funasr_local.text.phoneme_tokenizer import g2p_choices
from funasr_local.train.abs_espnet_model import AbsESPnetModel
from funasr_local.train.class_choices import ClassChoices
from funasr_local.train.trainer import Trainer
from funasr_local.utils.get_default_kwargs import get_default_kwargs
from funasr_local.utils.nested_dict_action import NestedDictAction
from funasr_local.utils.types import float_or_none
from funasr_local.utils.types import int_or_none
from funasr_local.utils.types import str2bool
from funasr_local.utils.types import str_or_none

from funasr_local.models.specaug.specaug import SpecAugLFR
from funasr_local.models.predictor.cif import CifPredictor, CifPredictorV2
from funasr_local.modules.subsampling import Conv1dSubsampling
from funasr_local.models.e2e_vad import E2EVadModel
from funasr_local.models.encoder.fsmn_encoder import FSMN

frontend_choices = ClassChoices(
    name="frontend",
    classes=dict(
        default=DefaultFrontend,
        sliding_window=SlidingWindow,
        s3prl=S3prlFrontend,
        fused=FusedFrontends,
        wav_frontend=WavFrontend,
        wav_frontend_online=WavFrontendOnline,
    ),
    type_check=AbsFrontend,
    default="default",
)
specaug_choices = ClassChoices(
    name="specaug",
    classes=dict(
        specaug=SpecAug,
        specaug_lfr=SpecAugLFR,
    ),
    type_check=AbsSpecAug,
    default=None,
    optional=True,
)
normalize_choices = ClassChoices(
    "normalize",
    classes=dict(
        global_mvn=GlobalMVN,
        utterance_mvn=UtteranceMVN,
    ),
    type_check=AbsNormalize,
    default=None,
    optional=True,
)
model_choices = ClassChoices(
    "model",
    classes=dict(
        e2evad=E2EVadModel,
    ),
    type_check=object,
    default="e2evad",
)

encoder_choices = ClassChoices(
    "encoder",
    classes=dict(
        fsmn=FSMN,
    ),
    type_check=torch.nn.Module,
    default="fsmn",
)


class VADTask(AbsTask):
    # If you need more than one optimizers, change this value
    num_optimizers: int = 1

    # Add variable objects configurations
    class_choices_list = [
        # --frontend and --frontend_conf
        frontend_choices,
        # --model and --model_conf
        model_choices,
    ]

    # If you need to modify train() or eval() procedures, change Trainer class here
    trainer = Trainer

    @classmethod
    def add_task_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(description="Task related")

        # NOTE(kamo): add_arguments(..., required=True) can't be used
        # to provide --print_config mode. Instead of it, do as
        # required = parser.get_default("required")
        # required += ["token_list"]

        group.add_argument(
            "--init",
            type=lambda x: str_or_none(x.lower()),
            default=None,
            help="The initialization method",
            choices=[
                "chainer",
                "xavier_uniform",
                "xavier_normal",
                "kaiming_uniform",
                "kaiming_normal",
                None,
            ],
        )

        group.add_argument(
            "--input_size",
            type=int_or_none,
            default=None,
            help="The number of input dimension of the feature",
        )

        group = parser.add_argument_group(description="Preprocess related")
        parser.add_argument(
            "--speech_volume_normalize",
            type=float_or_none,
            default=None,
            help="Scale the maximum amplitude to the given value.",
        )
        parser.add_argument(
            "--rir_scp",
            type=str_or_none,
            default=None,
            help="The file path of rir scp file.",
        )
        parser.add_argument(
            "--rir_apply_prob",
            type=float,
            default=1.0,
            help="THe probability for applying RIR convolution.",
        )
        parser.add_argument(
            "--cmvn_file",
            type=str_or_none,
            default=None,
            help="The file path of noise scp file.",
        )
        parser.add_argument(
            "--noise_scp",
            type=str_or_none,
            default=None,
            help="The file path of noise scp file.",
        )
        parser.add_argument(
            "--noise_apply_prob",
            type=float,
            default=1.0,
            help="The probability applying Noise adding.",
        )
        parser.add_argument(
            "--noise_db_range",
            type=str,
            default="13_15",
            help="The range of noise decibel level.",
        )

        for class_choices in cls.class_choices_list:
            # Append --<name> and --<name>_conf.
            # e.g. --encoder and --encoder_conf
            class_choices.add_arguments(group)

    @classmethod
    def build_collate_fn(
            cls, args: argparse.Namespace, train: bool
    ) -> Callable[
        [Collection[Tuple[str, Dict[str, np.ndarray]]]],
        Tuple[List[str], Dict[str, torch.Tensor]],
    ]:
        assert check_argument_types()
        # NOTE(kamo): int value = 0 is reserved by CTC-blank symbol
        return CommonCollateFn(float_pad_value=0.0, int_pad_value=-1)

    @classmethod
    def build_preprocess_fn(
            cls, args: argparse.Namespace, train: bool
    ) -> Optional[Callable[[str, Dict[str, np.array]], Dict[str, np.ndarray]]]:
        assert check_argument_types()
        # if args.use_preprocessor:
        #    retval = CommonPreprocessor(
        #        train=train,
        #        # NOTE(kamo): Check attribute existence for backward compatibility
        #        rir_scp=args.rir_scp if hasattr(args, "rir_scp") else None,
        #        rir_apply_prob=args.rir_apply_prob
        #        if hasattr(args, "rir_apply_prob")
        #        else 1.0,
        #        noise_scp=args.noise_scp if hasattr(args, "noise_scp") else None,
        #        noise_apply_prob=args.noise_apply_prob
        #        if hasattr(args, "noise_apply_prob")
        #        else 1.0,
        #        noise_db_range=args.noise_db_range
        #        if hasattr(args, "noise_db_range")
        #        else "13_15",
        #        speech_volume_normalize=args.speech_volume_normalize
        #        if hasattr(args, "rir_scp")
        #        else None,
        #    )
        # else:
        #    retval = None
        retval = None
        assert check_return_type(retval)
        return retval

    @classmethod
    def required_data_names(
            cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        if not inference:
            retval = ("speech", "text")
        else:
            # Recognition mode
            retval = ("speech",)
        return retval

    @classmethod
    def optional_data_names(
            cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        retval = ()
        assert check_return_type(retval)
        return retval

    @classmethod
    def build_model(cls, args: argparse.Namespace):
        assert check_argument_types()
        # 4. Encoder
        encoder_class = encoder_choices.get_class(args.encoder)
        encoder = encoder_class(**args.encoder_conf)

        # 5. Build model
        try:
            model_class = model_choices.get_class(args.model)
        except AttributeError:
            model_class = model_choices.get_class("e2evad")
        
        # 1. frontend
        if args.input_size is None:
            # Extract features in the model
            frontend_class = frontend_choices.get_class(args.frontend)
            if args.frontend == 'wav_frontend':
                frontend = frontend_class(cmvn_file=args.cmvn_file, **args.frontend_conf)
            else:
                frontend = frontend_class(**args.frontend_conf)
            input_size = frontend.output_size()
        else:
            # Give features from data-loader
            args.frontend = None
            args.frontend_conf = {}
            frontend = None
            input_size = args.input_size
        
        model = model_class(encoder=encoder, vad_post_args=args.vad_post_conf, frontend=frontend)

        return model

    # ~~~~~~~~~ The methods below are mainly used for inference ~~~~~~~~~
    @classmethod
    def build_model_from_file(
            cls,
            config_file: Union[Path, str] = None,
            model_file: Union[Path, str] = None,
            device: str = "cpu",
            cmvn_file: Union[Path, str] = None,
    ):
        """Build model from the files.

        This method is used for inference or fine-tuning.

        Args:
            config_file: The yaml file saved when training.
            model_file: The model file saved when training.
            device: Device type, "cpu", "cuda", or "cuda:N".

        """
        assert check_argument_types()
        if config_file is None:
            assert model_file is not None, (
                "The argument 'model_file' must be provided "
                "if the argument 'config_file' is not specified."
            )
            config_file = Path(model_file).parent / "config.yaml"
        else:
            config_file = Path(config_file)

        with config_file.open("r", encoding="utf-8") as f:
            args = yaml.safe_load(f)
        #if cmvn_file is not None:
        args["cmvn_file"] = cmvn_file
        args = argparse.Namespace(**args)
        model = cls.build_model(args)
        model.to(device)
        model_dict = dict()
        model_name_pth = None
        if model_file is not None:
            logging.info("model_file is {}".format(model_file))
            if device == "cuda":
                device = f"cuda:{torch.cuda.current_device()}"
            model_dir = os.path.dirname(model_file)
            model_name = os.path.basename(model_file)
            model_dict = torch.load(model_file, map_location=device)
        model.encoder.load_state_dict(model_dict)

        return model, args
