import json
import numpy as np

# Load results
with open('logs/eval_intent/predictions_3scenarios.json', 'r') as f:
    pred_results = json.load(f)

with open('logs/eval_intent/digital_twin_nav_mask_20260204_203700.json', 'r') as f:
    dt_results = json.load(f)

print('=== evaluate_intent_prediction.py ===')
samples = pred_results['samples']
ades = [s['ADE'] for s in samples if s['ADE'] < 50]  # filter outliers
print(f'Samples (ADE<50): {len(ades)}')
print(f'ADE: {np.mean(ades):.2f} ± {np.std(ades):.2f} m')

# Check first/last waypoint predictions
print('\nFirst 5 samples:')
for i, s in enumerate(samples[:5]):
    print(f"  Sample {i+1}: Pred first=[{s['predicted_waypoints'][0][0]:.2f}, {s['predicted_waypoints'][0][1]:.2f}], GT first=[{s['gt_waypoints'][0][0]:.2f}, {s['gt_waypoints'][0][1]:.2f}]")

print()
print('=== evaluate_intent_digital_twin.py ===')
samples_dt = dt_results['samples']
print(f'Samples: {len(samples_dt)}')
ades_dt = [s['ADE'] for s in samples_dt]
print(f'ADE: {np.mean(ades_dt):.2f} ± {np.std(ades_dt):.2f} m')

# Check first/last waypoint predictions  
print('\nFirst 5 samples:')
for i, s in enumerate(samples_dt[:5]):
    print(f"  Sample {i+1}: Pred first=[{s['predicted_waypoints'][0][0]:.2f}, {s['predicted_waypoints'][0][1]:.2f}], GT first=[{s['gt_waypoints'][0][0]:.2f}, {s['gt_waypoints'][0][1]:.2f}]")

# Analyze prediction patterns
print('\n' + '='*60)
print('Prediction Pattern Analysis')
print('='*60)

# For raw prediction method
print('\n[Raw Prediction Method - 3 scenarios]')
pred_x_first = [s['predicted_waypoints'][0][0] for s in samples if s['ADE'] < 50]
pred_y_first = [s['predicted_waypoints'][0][1] for s in samples if s['ADE'] < 50]
gt_x_first = [s['gt_waypoints'][0][0] for s in samples if s['ADE'] < 50]
gt_y_first = [s['gt_waypoints'][0][1] for s in samples if s['ADE'] < 50]

print(f'First waypoint (5m):')
print(f'  Pred X: {np.mean(pred_x_first):.2f} ± {np.std(pred_x_first):.2f} m')
print(f'  GT X:   {np.mean(gt_x_first):.2f} ± {np.std(gt_x_first):.2f} m')
print(f'  Pred Y: {np.mean(pred_y_first):.2f} ± {np.std(pred_y_first):.2f} m')
print(f'  GT Y:   {np.mean(gt_y_first):.2f} ± {np.std(gt_y_first):.2f} m')

pred_x_last = [s['predicted_waypoints'][-1][0] for s in samples if s['ADE'] < 50]
pred_y_last = [s['predicted_waypoints'][-1][1] for s in samples if s['ADE'] < 50]
gt_x_last = [s['gt_waypoints'][-1][0] for s in samples if s['ADE'] < 50]
gt_y_last = [s['gt_waypoints'][-1][1] for s in samples if s['ADE'] < 50]

print(f'Last waypoint (25m):')
print(f'  Pred X: {np.mean(pred_x_last):.2f} ± {np.std(pred_x_last):.2f} m')
print(f'  GT X:   {np.mean(gt_x_last):.2f} ± {np.std(gt_x_last):.2f} m')
print(f'  Pred Y: {np.mean(pred_y_last):.2f} ± {np.std(pred_y_last):.2f} m')
print(f'  GT Y:   {np.mean(gt_y_last):.2f} ± {np.std(gt_y_last):.2f} m')

# For digital twin method
print('\n[Digital Twin Method]')
pred_x_first = [s['predicted_waypoints'][0][0] for s in samples_dt]
pred_y_first = [s['predicted_waypoints'][0][1] for s in samples_dt]
gt_x_first = [s['gt_waypoints'][0][0] for s in samples_dt]
gt_y_first = [s['gt_waypoints'][0][1] for s in samples_dt]

print(f'First waypoint (5m):')
print(f'  Pred X: {np.mean(pred_x_first):.2f} ± {np.std(pred_x_first):.2f} m')
print(f'  GT X:   {np.mean(gt_x_first):.2f} ± {np.std(gt_x_first):.2f} m')
print(f'  Pred Y: {np.mean(pred_y_first):.2f} ± {np.std(pred_y_first):.2f} m')
print(f'  GT Y:   {np.mean(gt_y_first):.2f} ± {np.std(gt_y_first):.2f} m')

pred_x_last = [s['predicted_waypoints'][-1][0] for s in samples_dt]
pred_y_last = [s['predicted_waypoints'][-1][1] for s in samples_dt]
gt_x_last = [s['gt_waypoints'][-1][0] for s in samples_dt]
gt_y_last = [s['gt_waypoints'][-1][1] for s in samples_dt]

print(f'Last waypoint (25m):')
print(f'  Pred X: {np.mean(pred_x_last):.2f} ± {np.std(pred_x_last):.2f} m')
print(f'  GT X:   {np.mean(gt_x_last):.2f} ± {np.std(gt_x_last):.2f} m')
print(f'  Pred Y: {np.mean(pred_y_last):.2f} ± {np.std(pred_y_last):.2f} m')
print(f'  GT Y:   {np.mean(gt_y_last):.2f} ± {np.std(gt_y_last):.2f} m')
