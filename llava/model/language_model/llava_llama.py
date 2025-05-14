#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from llava.model.utils import save_attention, save_word_attention

class LlavaConfig(LlamaConfig):
    model_type = "llava"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig
    attention_gradients = []
    attention_weights = []

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        print("LlavaLlamaForCausalLM::init")
        self.model = LlavaLlamaModel(config)
        self.image_name = None

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
        self.inithook()
        

    def inithook(self):
        for layer in self.model.layers:
            # layer.self_attn.register_forward_hook(self.save_attention_weights)
            layer.self_attn.register_backward_hook(self.save_attention_gradients)
    

    def get_model(self):
        print("LlavaLlamaForCausalLM::get_model")
        return self.model
    
    def get_imagename(self):
        print("LlavaLlamaForCausalLM::get_imagename")
        return self.image_name
    
    def save_attention_gradients(self, module, grad_input, grad_output):
        self.attention_gradients.append(grad_output[0].detach())

    def save_attention_weights(self, module, input, output):
        self.attention_weights.append(output)     


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        image_name: Optional[str] = None
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        print("LlavaLlamaForCausalLM::forward")
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_ids, attention_mask, past_key_values, inputs_embeds, labels, img_index = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, images, image_name)


        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        print("Output attention : ", len(outputs.attentions), outputs.attentions[0].shape)

        if input_ids==None:
            self.image_token_start, self.image_token_len = img_index
            print("image_token_start, image_token_len: ", self.image_token_start, self.image_token_len)
            save_attention("decoder"+image_name, images, outputs.attentions, self.image_token_start, self.image_token_len)    
        else:
            input_ids = str(input_ids[0][0].detach().cpu().numpy())
            save_word_attention("decoder"+ input_ids +image_name, images, outputs.attentions, self.image_token_start, self.image_token_len)    

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        print("Logits : ", logits.shape)
        logits.backward()

        for grads in self.attention_gradients:
            print("Grads :", grads.shape)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
        print("Loss :", loss)    

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        print("LlavaLlamaForCausalLM::prepare_inputs_for_generation")
        self.image_name = kwargs.get("image_name")
        # print(kwargs.get("image_name"))
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
                "image_name": kwargs.get("image_name")
            }
        )
        return model_inputs

AutoConfig.register("llava", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
