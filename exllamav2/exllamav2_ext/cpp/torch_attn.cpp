#include "torch_attn.h"

torch::Tensor repeat_kv
(
    torch::Tensor states,
    int num_groups
)
{
    auto batch                  = states.size(0);
    auto num_key_value_heads    = states.size(1);
    auto len                    = states.size(2);
    auto head_dim               = states.size(3);

    auto expanded_states = states.unsqueeze(2).expand({batch, num_key_value_heads, num_groups, len, head_dim});
    auto reshaped_states = expanded_states.reshape({batch, num_key_value_heads * num_groups, len, head_dim});
    return reshaped_states;
}

torch::Tensor torch_attn
(
    torch::Tensor key_states,
    torch::Tensor value_states,
    torch::Tensor query_states,
    int past_len,
    int q_len,
    int batch_size,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    torch::Tensor attn_mask
)
{
    int num_kv_groups = num_heads / num_kv_heads;

    // Q: [batch_size, q_len,               num_heads, head_dim]
    // K: [batch_size, past_len + q_len, num_kv_heads, head_dim] (strided)
    // V: [batch_size, past_len + q_len, num_kv_heads, head_dim] (strided)

    // Torch attention

    query_states = query_states.transpose(1, 2);
    key_states = key_states.transpose(1, 2);
    value_states = value_states.transpose(1, 2);

    if (num_kv_groups > 1)
    {
        key_states = repeat_kv(key_states, num_kv_groups);
        // K: [batch_size, past_len + q_len, num_kv_heads * num_kv_groups = num_heads, head_dim] (strided)
    }
    key_states = key_states.transpose(2, 3);

    torch::Tensor attn_weights = torch::matmul(query_states, key_states);
    attn_weights *= (1.0f / std::sqrt(static_cast<double>(head_dim)));

    if (!attn_mask.is_meta()) attn_weights += attn_mask;

    attn_weights = torch::nn::functional::softmax(attn_weights, -1);

    if (num_heads != num_kv_heads)
    {
        value_states = repeat_kv(value_states, num_kv_groups);
        // V: [batch_size, past_len + q_len, num_kv_heads * num_kv_groups = num_heads, head_dim] (strided)
    }
    torch::Tensor attn_output = torch::matmul(attn_weights, value_states);

    attn_output = attn_output.transpose(1, 2);
    attn_output = attn_output.reshape({batch_size, q_len, num_heads * head_dim});

    return attn_output;
}