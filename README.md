# SimpleText Best of Labs in CLEF-2023: Scientific Text Simplification Using Multi-Prompt Minimum Bayes Risk Decoding

This repository includes a Jupyter Notebook (`clef2024-summary.ipynb`) that accompanies the paper. The notebook is designed to replicate the experiments discussed in the paper, particularly focusing on text simplification using multiple prompt strategies and Minimum Bayes Risk (MBR) decoding.

> [!NOTE]
> Note that we haven’t published the entire codebase for the experiments in the paper. If you have any questions or need further information, please contact us.

## Resources

- **Llama 3 Model:** The simplifications in this study are generated using the Llama 3 model. You can find more information about the model here: [Meta Llama 3 - 8B Instruct on Hugging Face](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct).
- **LENS Metric:** The evaluation metric used in the Minimum Bayes Risk decoding process is LENS. More details about LENS can be found in the official GitHub repository: [LENS GitHub Repository](https://github.com/Yao-Dou/LENS).

## Code from the Previous Paper (CLEF-2023)

The folder [prompts-2023](prompts-2023) contains the prompts used for the previous paper "UZH_Pandas at SimpleText@CLEF-2023: Alpaca LoRA 7B and LENS Model Selection for Scientific Literature Simplification". Note that the code for the fine-tuning of Alpaca LoRA 7B can be found in the [Alpaca LoRA 7B repository](https://github.com/tloen/alpaca-lora).

<details>
  <summary>Fine-tuning Parameters</summary>

  ```python
  train(
      base_model="chainyo/alpaca-lora-7b",
      data_path=filename,
      prompt_template_name=template,
      num_epochs=3,
      cutoff_len=512,
      batch_size=64,
      group_by_length=True,
      val_set_size=0.2,
      output_dir=config.data_dir(f"alpaca-lora-both-{template}"),
      lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
      lora_r=16,
      micro_batch_size=8,
  )
```
</details>

## Contributors

- [Andrianos Michail](https://www.cl.uzh.ch/de/about-us/people/team/compling/amichail.html) (andrianos.michail@cl.uzh.ch)
- [Pascal Severin Andermatt](https://www.ifi.uzh.ch/en/ddis/people/pandermatt.html) (pandermatt@ifi.uzh.ch)
- Tobias Fankhauser (tobias.fankhauser@uzh.ch)
