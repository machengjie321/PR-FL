
import matplotlib.pyplot as plt
import numpy as np
max_acc =[0.6415018315018315,0.6415018315018315, 0.6560317460317462, 0.6480463980463982, 0.6558730158730158, 0.6454822954822955, 0.6462759462759462, 0.6500854700854701, 0.6431746031746033, 0.6404151404151405, 0.6235775335775336, 0.617997557997558, 0.6282173382173383, 0.573943833943834]
max_npmi =[0.055869287487912235,0.055869287487912235, 0.04848020173365588, 0.04286419633912688, 0.04316669171395274, 0.046376306083142166, 0.04839196704975689, 0.037760546872064524, 0.06562529291609563, 0.05837298271954422, 0.054760466043475106, 0.031205540845559435, 0.05878822796293751, 0.022427538983528585]
max_td =[0.5215,0.5215, 0.513, 0.49249999999999994, 0.5325, 0.542, 0.508, 0.5145, 0.5485, 0.48749999999999993, 0.49599999999999994, 0.44949999999999996, 0.5095, 0.631]
#
# max_acc = [0.4351859419697589,0.4351859419697589, 0.43334695545566004, 0.42460155292194535, 0.43314262362076017, 0.4274213322435636, 0.4224765018389865, 0.440008173273396, 0.43363302002451987, 0.4231712300776462, 0.41855333060890887, 0.41532488761749087, 0.3720474049856969, 0.35210461789946873]
# max_npmi = [0.07897381731098824,0.07897381731098824, 0.04338192498046494, 0.04535100549492724, 0.052272607405530656, 0.053981179154414125, 0.061677581500359534, 0.0704645355874615, 0.06593487915732098, 0.03845331490013492, 0.038160677393151876, 0.052003835373439236, 0.05613969277163424, 0.01690064206610995]
# max_td = [0.8769999999999999,0.8769999999999999, 0.863, 0.8955000000000002, 0.848, 0.8930000000000001, 0.8850000000000001, 0.8445000000000001, 0.8530000000000001, 0.8365, 0.8960000000000001, 0.8405, 0.8385, 0.7735000000000001]
max_acc_normal = []
max_acc_fast = []
max_coherence_normal = []
max_coherence_fast = []
max_diversity_normal = []
max_diversity_fast = []
for i in range(len(max_acc)):
    if i%2 == 0:
        max_acc_normal.append(max_acc[i])
        max_coherence_normal.append(max_npmi[i])
        max_diversity_normal.append(max_td[i])
    else:
        max_acc_fast.append(max_acc[i])
        max_coherence_fast.append(max_npmi[i])
        max_diversity_fast.append(max_td[i])


x_lim = ['1.0','0.8','0.6','0.4','0.2','0.1','0.01']

x = np.arange(len(max_acc_fast))  # 横坐标范围
font2 = {'family' : 'Times New Roman',
         'weight' : 'normal',
         'size'  : 15,
         }

plt.figure()
total_width, n = 0.8, 2  # 柱状图总宽度，有几组数据
width = total_width / n  # 单个柱状图的宽度
x1 = x - width / 2  # 第一组数据柱状图横坐标起始位置
x2 = x1 + width  # 第二组数据柱状图横坐标起始位置
plt.bar(x1, max_acc_normal, width=width, label='Normal')
plt.bar(x2, max_acc_fast, width=width, label='Fast')
plt.xlabel("Target density",font2)
plt.ylim([0.50,0.70])
plt.ylabel("Test Accuracy",font2)
plt.grid(linestyle="--", color='black', lw='0.05', alpha=0.5)
plt.xticks(x, x_lim)   # 用星期几替换横坐标x的值
plt.legend()
plt.savefig('max_acc.svg',dpi=650)
plt.show()

plt.figure()
total_width, n = 0.8, 2  # 柱状图总宽度，有几组数据
width = total_width / n  # 单个柱状图的宽度
x1 = x - width / 2  # 第一组数据柱状图横坐标起始位置
x2 = x1 + width  # 第二组数据柱状图横坐标起始位置
plt.bar(x1, max_coherence_normal, width=width, label='Normal')
plt.bar(x2, max_coherence_fast, width=width, label='Fast')
plt.xlabel("Target density",font2)
plt.ylabel("Test Coherence",font2)
plt.grid(linestyle="--", color='black', lw='0.05', alpha=0.5)
plt.xticks(x, x_lim)   # 用星期几替换横坐标x的值
plt.ylim([0,0.08])
plt.legend()
plt.savefig('max_coherence.svg',dpi=750)
plt.show()

plt.figure()
total_width, n = 0.8, 2  # 柱状图总宽度，有几组数据
width = total_width / n  # 单个柱状图的宽度
x1 = x - width / 2  # 第一组数据柱状图横坐标起始位置
x2 = x1 + width  # 第二组数据柱状图横坐标起始位置
plt.bar(x1, max_diversity_normal, width=width, label='Normal')
plt.bar(x2, max_diversity_fast, width=width, label='Fast')
#plt.ylim([0.7,0.95])
plt.ylim([0.0,0.7])
plt.xlabel("Target density",font2)
plt.ylabel("Test Diversity",font2)
plt.grid(linestyle="--", color='black', lw='0.05', alpha=0.5)
plt.xticks(x, x_lim)   # 用星期几替换横坐标x的值
plt.legend()
plt.savefig('max_diversity.svg',dpi=750)
plt.show()