from transformers import AutoConfig

import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch.nn.functional as F
import os
import torch

def auto_upgrade(config):
    cfg = AutoConfig.from_pretrained(config)
    if 'llava' in config and 'llava' not in cfg.model_type:
        assert cfg.model_type == 'llama'
        print("You are using newer LLaVA code base, while the checkpoint of v0 is from older code base.")
        print("You must upgrade the checkpoint to the new code base (this can be done automatically).")
        confirm = input("Please confirm that you want to upgrade the checkpoint. [Y/N]")
        if confirm.lower() in ["y", "yes"]:
            print("Upgrading checkpoint...")
            assert len(cfg.architectures) == 1
            setattr(cfg.__class__, "model_type", "llava")
            cfg.architectures[0] = 'LlavaLlamaForCausalLM'
            cfg.save_pretrained(config)
            print("Checkpoint upgraded.")
        else:
            print("Checkpoint upgrade aborted.")
            exit(1)


def save_attention_grid(attentions, original_image, filename="attention_grid", dirname="/data/mariammaa/llava_rad/attention_grids/",image_token_start=None, image_token_len=None):
    num_heads = attentions.shape[1]  # [batch, heads, seq_len, seq_len]
    if num_heads>20:
        n_cols = 8
    else:
        n_cols = 4    
    n_rows = num_heads/n_cols
    fig, axes = plt.subplots(int(n_rows), n_cols, figsize=(20, 20))
    for k in range(num_heads):
        i = int(k/n_cols)
        j = int(k%n_cols)
        ax = axes[i, j]
        attn = attentions[0, k].detach().cpu()
        if image_token_start!=None and image_token_len!=None:
                attn = attn[image_token_start:image_token_start+image_token_len, image_token_start:image_token_start+image_token_len]

        attn = attn.to(dtype=torch.float32).contiguous(memory_format=torch.contiguous_format)
        attn = F.interpolate(attn.unsqueeze(0).unsqueeze(0), size=(518, 518), mode='bilinear', align_corners=False).squeeze()  # Back to (336, 336)
        attn_normalized = (attn - attn.min()) / (attn.max() - attn.min())
        vis = show_cam_on_image(original_image, attn_normalized)
        vis =  np.uint8(255 * vis)
        vis = cv2.cvtColor(np.array(vis), cv2.COLOR_BGR2RGB)
        ax.imshow(vis)
        ax.set_title(f"H{k}")
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(dirname, filename))
    plt.close()
    return
    

# create heatmap from mask on image
def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return cam    


def save_attention(image_name, images, attention_maps, image_token_start=None, image_token_len=None):
    image_name = image_name.split(".")[0]
    image_name = image_name.replace("/","_")
    save_dir = "/data/mariammaa/llava_rad/attention_maps/"
    save_dir_attngrid = "/data/mariammaa/llava_rad/attention_grids/"
    original_image = images.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    original_image = (original_image - original_image.min()) / (original_image.max() - original_image.min())
    for i in range(len(attention_maps)):
        for j in range(attention_maps[i].shape[1]):
            attn_tensor =  attention_maps[i][0,j].detach().cpu()  
            if image_token_start!=None and image_token_len!=None:
                attn_tensor = attn_tensor[image_token_start:image_token_start+image_token_len, image_token_start:image_token_start+image_token_len]

            attn_normalized = (attn_tensor - attn_tensor.min()) / (attn_tensor.max() - attn_tensor.min())
                
            save_img = os.path.join(save_dir,image_name + 'layer' + str(i) + 'head' + str(j) + '.png')
 
            attn_normalized = attn_normalized.to(dtype=torch.float32).contiguous(memory_format=torch.contiguous_format)
            # resized_att_normalized = F.interpolate(attn_normalized.unsqueeze(0).unsqueeze(0), size=(518, 518), mode='bilinear', align_corners=False).squeeze()  # Back to (336, 336)

            # vis = show_cam_on_image(original_image, resized_att_normalized)
            # # cv2.imwrite(save_img, cv2.cvtColor(attn_colored, cv2.COLOR_BGR2RGB))
            # vis =  np.uint8(255 * vis)
            # vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
            # cv2.imwrite(save_img, cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))

        save_attention_grid(attention_maps[i], original_image, filename=image_name+"_grid"+str(i)+".png", image_token_start=image_token_start, image_token_len=image_token_len)

    if image_token_start!=None and image_token_len!=None:
        new_attention_maps = []
        for attention in attention_maps:
            attention = attention[:,:, image_token_start:image_token_start+image_token_len, image_token_start:image_token_start+image_token_len]        
            new_attention_maps.append(attention)
            attn_rollout = attention_rollout(new_attention_maps)
    else:          
        attn_rollout = attention_rollout(attention_maps)  
    print("Attention rollout :", attn_rollout.shape)
    resized_rollout_att_normalized = F.interpolate(attn_rollout.unsqueeze(0), size=(518, 518), mode='bilinear', align_corners=False).squeeze()  # Back to (336, 336)
    vis = show_cam_on_image(original_image, resized_rollout_att_normalized.detach().cpu())
    # cv2.imwrite(save_img, cv2.cvtColor(attn_colored, cv2.COLOR_BGR2RGB))
    vis =  np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    rollout_img_name = os.path.join(save_dir_attngrid, 'rollout_sumnorm'+image_name + ".png")
    cv2.imwrite(rollout_img_name, cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))


def attention_rollout(attentions):
    # Initialize rollout with identity matrix
    rollout = torch.eye(attentions[0].size(-1)).to(attentions[0].device)
    # Multiply attention maps layer by layer
    for attention in attentions:
        # print("Attention :", attention.shape)
        attention_heads_fused = attention.sum(dim=1) # Average attention across heads
        # print("Attention :", attention_heads_fused.shape)
        # attention_heads_fused += torch.eye(attention_heads_fused.size(-1)).to(attention_heads_fused.device) # A + I
        attention_heads_fused /= attention_heads_fused.sum(dim=-1, keepdim=True) # Normalizing A
        # print("Attention :", attention_heads_fused.shape)
        attention_heads_fused = attention_heads_fused.to(dtype=torch.float32).contiguous(memory_format=torch.contiguous_format)
        rollout = torch.matmul(rollout, attention_heads_fused) # Multiplication

    rollout = (rollout - rollout.min()) / (rollout.max() - rollout.min())
    return rollout
    

def save_word_attention(image_name, images, attention_maps, image_token_start=None, image_token_len=None): 
    image_name = image_name.split(".")[0]
    image_name = image_name.replace("/","_")
    save_dir = "/data/mariammaa/llava_rad/attention_maps/"
    save_dir_attngrid = "/data/mariammaa/llava_rad/attention_grids/"
    # original_image = images.squeeze(0).permute(1, 2, 0).detach().cpu()
    original_image = images
    original_image = F.interpolate(original_image, size=(100, 100), mode='bilinear', align_corners=False).squeeze()  # Back to (336, 336)
    original_image = (original_image - original_image.min()) / (original_image.max() - original_image.min())
    original_image = original_image.permute(1, 2, 0).detach().cpu().numpy()
    
    
    for i in range(len(attention_maps)):
        num_heads = attention_maps[i].shape[1]  # [batch, heads, seq_len, seq_len]
        n_cols = 6
        n_rows = int(num_heads/n_cols) + 1
        fig, axes = plt.subplots(int(n_rows), n_cols, figsize=(20, 20))

        for j in range(attention_maps[i].shape[1]):
            attn_tensor =  attention_maps[i][0,j].detach().cpu()  
                
            attn_tensor = attn_tensor[:,image_token_start:image_token_start+image_token_len]

            attn_normalized = (attn_tensor - attn_tensor.min()) / (attn_tensor.max() - attn_tensor.min())
                   
            save_img = os.path.join(save_dir,'word' + image_name + 'layer' + str(i) + 'head' + str(j) + '.png')

            # print("resized_att_normalized : ", attn_normalized.shape)
            attn_normalized = attn_normalized.reshape(37, 37)
            attn_normalized = attn_normalized.to(dtype=torch.float32).contiguous(memory_format=torch.contiguous_format)
            resized_att_normalized = F.interpolate(attn_normalized.unsqueeze(0).unsqueeze(0), size=(100, 100), mode='bilinear', align_corners=False).squeeze()  # Back to (336, 336)

            # print("resized_att_normalized : ", resized_att_normalized.shape)

            vis = show_cam_on_image(original_image, resized_att_normalized)
            # cv2.imwrite(save_img, cv2.cvtColor(attn_colored, cv2.COLOR_BGR2RGB))
            vis =  np.uint8(255 * vis)
            vis = cv2.cvtColor(np.array(vis), cv2.COLOR_BGR2RGB)
            # cv2.imwrite(save_img, cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))

            row = int(j/n_cols)
            col = int(j%n_cols)
            ax = axes[row, col]
            ax.imshow(vis)
            ax.set_title(f"H{j}")
            ax.axis('off')

        save_img_name = os.path.join(save_dir_attngrid, 'wordgrid' + image_name + 'layer' + str(i) + '.png')
        plt.tight_layout()
        plt.savefig(save_img_name)
        plt.close()

    new_attention_maps = []
    for attention in attention_maps:    
        attention = attention[:,:,:,image_token_start:image_token_start+image_token_len]    
        attention = attention.reshape(1,32,37, 37)
        new_attention_maps.append(attention)
    attn_rollout = attention_rollout(new_attention_maps)    
    attn_rollout = attn_rollout.to(dtype=torch.float32).contiguous(memory_format=torch.contiguous_format)
    attn_rollout = F.interpolate(attn_rollout.unsqueeze(0), size=(100, 100), mode='bilinear', align_corners=False).squeeze()  # Back to (336, 336)
    print("attention rollout: ", attn_rollout.shape)
    vis = show_cam_on_image(original_image, attn_rollout.detach().cpu())
    # cv2.imwrite(save_img, cv2.cvtColor(attn_colored, cv2.COLOR_BGR2RGB))
    vis =  np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR) 
    rollout_img_name = os.path.join(save_dir_attngrid, 'rollout_word_sumnorm'+image_name + ".png")
    cv2.imwrite(rollout_img_name, cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))

    return