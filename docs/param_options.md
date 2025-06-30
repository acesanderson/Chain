### Load Time Options
1. **numa (bool)**: This option controls whether to use Non-Uniform Memory Access (NUMA) optimizations. NUMA systems allocate memory more efficiently based on the proximity of CPU cores to memory nodes, which can improve performance for certain workloads.
2. **num_ctx (int)**: This specifies the maximum number of tokens (text chunks) that the model can consider at once during generation. It affects the context length and the size of the input sequence the model can handle.
3. **num_batch (int)**: This sets the number of batches to use for generating text. A larger batch size can be more efficient but requires more memory, while a smaller batch size might be slower but uses less memory.
4. **num_gpu (int)**: Specifies the number of GPUs to use for running the model. If there are multiple GPUs available, this parameter allows you to specify which ones to utilize.
5. **main_gpu (int)**: This option is used to specify which GPU should be considered as the main GPU for certain tasks or operations. It helps in managing GPU resources and can improve performance by focusing computations on a specific GPU.
6. **low_vram (bool)**: When set to `True`, this parameter suggests that the model should operate using less video RAM, which is useful for running the model on GPUs with limited memory.
7. **f16_kv (bool)**: This option uses half-precision floating point numbers for storing key and value matrices in the attention mechanism, which can save memory but might slightly impact performance.
8. **logits_all (bool)**: When set to `True`, this parameter includes all logits in the output, rather than just the last token's logits. This can be useful for tasks requiring more detailed probability distributions over sequences.
9. **vocab_only (bool)**: Restricts the model to only use the vocabulary provided during training, which might be useful in controlled environments or when dealing with specific datasets.
10. **use_mmap (bool)**: Enables memory-mapped file access for faster data loading and sharing between processes. This can improve performance, especially when working with large models on systems with multiple CPUs.
11. **use_mlock (bool)**: Locks the model in RAM to avoid swapping out to disk, which is useful for maintaining fast access times but might require more memory.
12. **embedding_only (bool)**: This option allows the model to run only in embedding mode, where it generates vector embeddings rather than full text sequences. Useful for tasks requiring high-dimensional data representations.
13. **num_thread (int)**: Specifies the number of threads to use during parallel processing. More threads can speed up certain operations but will consume more CPU resources and might not always be beneficial depending on system load.

### Runtime Options

1. **num_keep (int)**: This parameter controls how many tokens are kept in the context as the model generates new text. It helps maintain a stable context for coherent output, especially important in creative writing or conversational settings.
2. **seed (int)**: A seed value used to initialize the random number generator for repeatable randomness during generation. Useful for testing and debugging to ensure consistent results from run to run.
3. **num_predict (int)**: Specifies how many tokens should be generated in total before stopping. This is a hard limit on the length of the output sequence.
4. **top_k (int)**: Restricts the generation to the top K most probable tokens, which can control the diversity and randomness of the output. Higher values make the output more random, while lower values restrict it.
5. **top_p (float)**: Implements nucleus sampling where only a subset of tokens with the highest probabilities are considered for the next token selection based on their cumulative probability. This can balance diversity and coherence in the generated text.
6. **tfs_z (float)**: Tail Free Sampling parameter that helps reduce the likelihood of less probable tokens by applying a multiplicative factor to them, enhancing the quality of high-probability tokens in the output.
7. **typical_p (float)**: Similar to top_p but uses typicality sampling to adjust probabilities based on their typicality scores, which can help balance diversity and coherence more precisely than both top_p and tfs_z individually.
8. **repeat_last_n (int)**: This parameter prevents the model from repeating the last N tokens too frequently, helping to avoid repetitive outputs by limiting the influence of recent tokens on future choices.
9. **temperature (float)**: Controls the randomness of predictions. Lower values make the output more deterministic and focused, while higher values increase randomness and allow for more diverse outcomes.
10. **repeat_penalty (float)**: Applies a penalty to tokens that have been recently generated, which can help reduce repetitive patterns by discouraging the model from selecting those tokens again too soon.
11. **presence_penalty (float)**: Penalizes new lines or sections of text based on their presence in the current context, which can encourage more focused and coherent output.
12. **frequency_penalty (float)**: Similar to presence penalty but penalizes tokens based on how frequently they appear, which can help reduce repetitive phrases by discouraging commonly used words or phrases.
13. **mirostat (int)**: Implements Mirostat sampling control for more interpretable and controllable text generation. It helps maintain a constant level of surprise across the generated text.
14. **mirostat_tau (float)**: Sets the target expected cross-entropy for Mirostat sampling, which controls how closely the model follows the original intended meaning or style in the generated output.
15. **mirostat_eta (float)**: Adjusts the step size of the parameter update in Mirostat sampling to fine-tune the balance between following the target and exploring other possibilities during generation.
16. **penalize_newline (bool)**: Adds a penalty for newlines, which can be useful in environments where excessive line breaks are undesirable or could affect formatting constraints.
17. **stop (Sequence[str])**: A sequence of strings that defines the stopping conditions for the generation process. When any of these sequences appear in the generated text, the model stops generating further tokens. This is useful for controlling the output format or ending a conversation gracefully.
