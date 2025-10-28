"""GPT-4 based semantic judge from Chao et al. 2024.

@article{chao2024jailbreaking,
  title={Jailbreaking black box large language models in twenty queries},
  author={Chao, Patrick and Robey, Alexander and Dobriban, Edgar and Hassani, Hamed and Pappas, George J and Wong, Eric},
  journal={arXiv preprint arXiv:2310.08419},
  year={2024}
}


"""

from .adaptive_attacks import AdaptiveAttacksJudge


class PAIRJudge(AdaptiveAttacksJudge):
    """Judge for PAIR (Prompt Automatic Iterative Refinement).

    This judge uses the same implementation as AdaptiveAttacksJudge,
    as both use the same GPT-4 based semantic evaluation approach.
    """

    pass
