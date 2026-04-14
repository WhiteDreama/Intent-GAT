import pickle
p='metadrive/assets/waymo/sd_training.tfrecord-00000-of-01000_2a1e44d405a6833f.pkl'
with open(p,'rb') as f:
    obj=pickle.load(f)
print(type(obj))
if isinstance(obj, dict):
    print('keys:', list(obj.keys())[:50])
    if 'tracks' in obj:
        tracks = obj['tracks']
        print('tracks type:', type(tracks), 'len:', len(tracks) if hasattr(tracks,'__len__') else 'unknown')
        if hasattr(tracks, '__len__') and len(tracks)>0:
            first_key = next(iter(tracks.keys()))
            t0 = tracks[first_key]
            print('track0 type:', type(t0))
            try:
                print('track0 keys:', list(t0.keys())[:50])
                if 'state' in t0:
                    st = t0['state']
                    print('state type:', type(st))
                    try:
                        print('state keys:', list(st.keys())[:50])
                        if 'position' in st:
                                pos = st['position']
                                print('position type:', type(pos), 'shape' , getattr(pos,'shape',None))
                                if hasattr(pos,'shape') and len(pos.shape)==2:
                                    print('first pos sample:', pos[0])
                        if 'heading' in st:
                            print('heading type:', type(st['heading']))
                        if 'velocity' in st:
                            print('velocity type:', type(st['velocity']), 'len', len(st['velocity']) if hasattr(st['velocity'],'__len__') else 'scalar')
                    except Exception:
                        pass
            except Exception:
                pass
elif isinstance(obj, list):
    print('len list', len(obj))
    print(type(obj[0]))
    if isinstance(obj[0], dict):
        print('sample keys:', list(obj[0].keys())[:50])
else:
    print('repr:', repr(obj)[:500])
