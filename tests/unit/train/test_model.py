import pytest
import torch

from prime_rl.trainer.config import AttnImplementation, ModelConfig
from prime_rl.trainer.model import get_model

BS = 1
SEQ_LEN = 8

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.filterwarnings("ignore:torch.get_autocast_gpu_dtype\\(\\) is deprecated:DeprecationWarning"),
]


@pytest.fixture(params=["sdpa", "flash_attention_2"])
def attn(request) -> AttnImplementation:
    """
    Fixture to test different attention implementations.
    """
    try:
        # ruff: noqa: F401
        import flash_attn
    except ImportError:
        pytest.skip("Flash Attention not available")
    return request.param


@pytest.fixture
def model(attn):
    config = ModelConfig(name="Qwen/Qwen3-0.6B", attn=attn)
    return get_model(config)


def test_model_to_gpu(model):
    model = model.to("cuda")


def test_model_forward(model):
    model = model.to("cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        inputs_ids = torch.randint(0, 100, (BS, SEQ_LEN)).to("cuda")
        outputs = model(input_ids=inputs_ids).logits

        assert outputs.shape == (BS, SEQ_LEN, model.config.vocab_size)


def test_model_with_position_ids(model):
    model = model.to("cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        inputs_ids = torch.randint(0, 100, (BS, SEQ_LEN)).to("cuda")
        position_ids = torch.arange(SEQ_LEN).unsqueeze(0).repeat(BS, 1).to("cuda")

        outputs = model(input_ids=inputs_ids, position_ids=position_ids).logits

        assert outputs.shape == (BS, SEQ_LEN, model.config.vocab_size)


@pytest.mark.skip(reason="Sequence packing for Qwen not working.")
@pytest.mark.parametrize("correct_position_ids", [True, False])
def test_model_with_sequence_packing(model, correct_position_ids):
    """
    The goal of this test is to check that the sequence packing works correctly.

    The idea is that is to check that the logits is the same when doing

    [B, seq]  and doing [1, B*seq] with the proper masking.

    """
    if model.config.attn != "flash_attention_2":
        pytest.skip("Test only works with flash attention")

    model = model.to("cuda")
    inputs = [0, 1, 2, 3]

    with torch.autocast("cuda", dtype=torch.bfloat16):
        inputs_ids = torch.Tensor(inputs).repeat(1, 1).int().to("cuda")
        output_base = model(input_ids=inputs_ids).logits

        assert output_base.shape == (1, len(inputs), model.config.vocab_size)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        inputs_ids = torch.Tensor(inputs + inputs).repeat(1, 1).int().to("cuda")
        if correct_position_ids:
            position_ids = torch.Tensor([0, 1, 2, 3, 0, 1, 2, 3]).repeat(1, 1).int().to("cuda")
            # should work
        else:
            position_ids = torch.Tensor([0, 1, 2, 3, 4, 5, 6, 7]).repeat(1, 1).int().to("cuda")
            # should fail
        outputs_packed = model(input_ids=inputs_ids, position_ids=position_ids).logits

        assert outputs_packed.shape == (1, 2 * len(inputs), model.config.vocab_size)

    output_packed_left = outputs_packed[:, : len(inputs), :]
    output_packed_right = outputs_packed[:, len(inputs) :, :]

    assert output_packed_left.shape == output_base.shape == output_packed_right.shape

    if correct_position_ids:
        torch.testing.assert_close(output_packed_left, output_base)
        torch.testing.assert_close(output_packed_right, output_base)
    else:
        torch.testing.assert_close(output_packed_left, output_base)
        with pytest.raises(AssertionError):
            torch.testing.assert_close(output_packed_right, output_base)


def test_moe_custom_impl():
    config = ModelConfig(name="Jackmin108/glm-0.5B", attn="sdpa", impl="custom", moe_use_grouped_mm=False)
    model = get_model(config)
    model = model.to("cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        inputs_ids = torch.randint(0, 100, (BS, SEQ_LEN)).to("cuda")
        outputs = model(input_ids=inputs_ids).logits

        assert outputs.shape == (BS, SEQ_LEN, model.config.vocab_size)


@pytest.mark.skip(reason="need special token for meta stuff in ci")
@pytest.mark.parametrize("model_name", ["meta-llama/Llama-3.2-1B-Instruct"])
def test_model_forward_custom_impl(model_name):
    config = ModelConfig(name=model_name, impl="custom", attn="sdpa")
    model = get_model(config)
    model = model.to("cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        inputs_ids = torch.randint(0, 100, (BS, SEQ_LEN)).to("cuda")
        outputs = model(input_ids=inputs_ids).logits

        assert outputs.shape == (BS, SEQ_LEN, model.config.vocab_size)
