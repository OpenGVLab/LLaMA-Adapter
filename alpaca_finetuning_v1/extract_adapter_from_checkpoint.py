import torch



model = torch.load('./checkpoint/checkpoint-4.pth', map_location='cpu')
new_model = dict()
weight_list = ['layers.' + str(i) + '.attention.gate' for i in range(32)]
old_weight_list = ['layers.' + str(i) + '.attention.gate' for i in range(32)]
weight_list = weight_list + ['adapter_query.weight']

print(weight_list)
print(model['model']['adapter_query.weight'].shape)

for i in range(len(weight_list)):
        new_model[weight_list[i]] = model['model'][weight_list[i]]

torch.save(new_model, 'adapter_adapter_len10_layer30_epoch5.pth')   
