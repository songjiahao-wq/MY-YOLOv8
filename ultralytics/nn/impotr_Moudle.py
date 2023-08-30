#add ATT
from add.cv_attention import GAM_Attention
from add.cv_attention.EffectiveSE import EffectiveSEModule
from ultralytics.nn.moudles_add import BiLevelRoutingAttention, AttentionLePE, Attention
from ultralytics.nn.fighting_model.backbone.HorNet import HorNet
from ultralytics.nn.fighting_model.backbone.convnextv2 import convnextv2_att
from ultralytics.nn.add_models.my_attention import *
from ultralytics.nn.fighting_model.backbone.repghost import RepGhostBottleneck




from ultralytics.nn.fighting_model.conv.ODConv import ODConv2d
from ultralytics.nn.fighting_model.conv.MBConv import MBConvBlock
from ultralytics.nn.fighting_model.conv.CondConv import CondConv
from ultralytics.nn.fighting_model.conv.DynamicConv import DynamicConv
from ultralytics.nn.fighting_model.conv.HorNet import gnconv  # use: gnconv(dim), c1 = c2