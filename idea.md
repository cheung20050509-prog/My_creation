/root/autodl-tmp/ITHP_MODS_CyIN_Ect/My_creation 是我的要投Neurips的作品

/root/autodl-tmp/ITHP_MODS_CyIN_Ect/My_creation以/root/autodl-tmp/ITHP_MODS_CyIN_Ect/original_ITHP 为框架，参考/root/autodl-tmp/ITHP_MODS_CyIN_Ect/ITHP_MODS和/root/autodl-tmp/ITHP_MODS_CyIN_Ect/CyIN的方法

/root/autodl-tmp/ITHP_MODS_CyIN_Ect/original_ITHP的源码是作者的源码。/root/autodl-tmp/ITHP_MODS_CyIN_Ect/ITHP_MODS和/root/autodl-tmp/ITHP_MODS_CyIN_Ect/CyIN的源码是我复现的源码，有错误可能，不要盲目参考。

/root/autodl-tmp/ITHP_MODS_CyIN_Ect/original_ITHP/ITHP.pdf /root/autodl-tmp/ITHP_MODS_CyIN_Ect/original_ITHP/ITHP.md /root/autodl-tmp/ITHP_MODS_CyIN_Ect/ITHP_MODS/MODS.pdf /root/autodl-tmp/ITHP_MODS_CyIN_Ect/ITHP_MODS/MODS.md /root/autodl-tmp/ITHP_MODS_CyIN_Ect/CyIN/CyIN_marker.md /root/autodl-tmp/ITHP_MODS_CyIN_Ect/CyIN/CyIN.pdf /root/autodl-tmp/ITHP_MODS_CyIN_Ect/CyIN/CyIN.tex
这些是论文，都是顶会的


下面是我的一些idea，尝试了实现

进入MSelector的特征，前面去掉了噪声，但不一定任务相关，参考CyIN的token-level IB那里，对这里特征做重参数采样。但是前面非文本模态会经过GDC也是做了噪声去除，感觉加入完整的IB可能会冗余


在现有技术上改进 比如怎么更好选择primary modality之类的 把ithp的ib用法改进

把cyin的合适的思想用在改进ithp的ib用法

/root/autodl-tmp/ITHP_MODS_CyIN_Ect/ITHP_MODS和/root/autodl-tmp/ITHP_MODS_CyIN_Ect/CyIN的源码是我复现的源码，有错误可能，不要盲目参考。

/root/autodl-tmp/ITHP_MODS_CyIN_Ect/ITHP_MODS和/root/autodl-tmp/ITHP_MODS_CyIN_Ect/CyIN的源码是我复现的源码，有错误可能，不要盲目参考。