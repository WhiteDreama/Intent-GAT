import pickle
p='marl_project/data/waymo_hard_buckets.pkl'
with open(p,'rb') as f:
    b=pickle.load(f)
for k,v in b.items():
    print('Bucket',k,'count',len(v))
    if len(v)>0:
        s=v[0]
        print('gt_trajectory_xy shape', type(s['gt_trajectory_xy']), getattr(s['gt_trajectory_xy'],'shape',None))
        print('gt_yaw shape', type(s['gt_yaw']), getattr(s['gt_yaw'],'shape',None))
        print('gt_velocity_ms shape', type(s['gt_velocity_ms']), getattr(s['gt_velocity_ms'],'shape',None))
        print('meta',s.get('meta'))
        break
