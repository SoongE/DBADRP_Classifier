import os

import pandas
import pandas as pd
import torch
from hydra import initialize, compose
from matplotlib import pyplot as plt
from torchmetrics.utilities.compute import auc

from data.loader import get_kfold_dataloader3d
from initial_setting import get_instance
from utils.calculate import margin_of_error


def inference(run, test_loader, model, criterion, wrong_item):
    acc_list, metrics, wrong_items = run.test(test_loader, model, criterion, wrong_item=wrong_item)
    value, moe = margin_of_error(acc_list)
    return metrics, value, moe, wrong_items


def main_worker(cfg, device, loader, write_wrong_predict_items, weight_files):
    weight_files = os.path.join('weights', weight_files)
    model, criterion, run = get_instance(cfg, device)

    print(f'Load {weight_files}')
    model.load_state_dict(torch.load(weight_files)['state_dict'])
    model.eval()

    metrics, value, moe, wrong_items = inference(run, loader, model, criterion, wrong_item=write_wrong_predict_items)

    if write_wrong_predict_items:
        log = _print(cfg.backbone, metrics)
        with open(f'wrong_predicts_{cfg.backbone}.txt', 'w') as f:
            f.write(log + '\n')
            f.write('path | pred | label \n')
            for path, pred, label in zip(*wrong_items):
                f.write(f'{path} | {pred} | {label} \n')
            f.write('\n\n')
    return metrics


def _print(backbone, metrics):
    log = f'[{backbone}] '
    for k, v in metrics.items():
        log += f'{k}:{v:.6f} | '
    return log[:-3]


def make_roc_curve_from_klist(roc_list, filename):
    fprs = [torch.zeros_like(roc_list[0][0][0]), torch.zeros_like(roc_list[0][0][1]),
            torch.zeros_like(roc_list[0][0][2])]
    tprs = [torch.zeros_like(roc_list[0][1][0]), torch.zeros_like(roc_list[0][1][1]),
            torch.zeros_like(roc_list[0][1][2])]

    for item in roc_list:
        for i in range(3):
            fprs[i] += item[0][i]
            tprs[i] += item[1][i]

    for i in range(3):
        fprs[i] = fprs[i] / len(roc_list)
        tprs[i] = tprs[i] / len(roc_list)

    colors = ["darkred", "darkorange", "cornflowerblue", 'darkgreen']
    labels = ["class normal", "class HGSBO", "class LGSBO", "average"]

    # For macro average
    fprs.append(torch.stack(fprs).mean(dim=0))
    tprs.append(torch.stack(tprs).mean(dim=0))
    # area = [0.97, 0.73, 0.88, 0.86]

    lw = 2
    for fpr, tpr, color, label in zip(fprs, tprs, colors, labels):
        fpr = fpr.detach().cpu()
        tpr = tpr.detach().cpu()
        area = str(auc(fpr, tpr).item())[:4]
        plt.plot(
            fpr, tpr, color=color, lw=lw,
            label=f"ROC curve of {label} (area = {area})",
        )

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    # plt.title("Some extension of Receiver operating characteristic to multiclass")
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(f'ROC_{filename}.svg', format='svg', dpi=300)
    plt.show()


def main():
    with initialize('configs'):
        cfg = compose('config', overrides=['train.batch_size=16', 'gpus=[0]', 'dataset=asbo-k'])
    write_wrong_predict_items = False
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in cfg.gpus)
    device = torch.device(f'cuda') if torch.cuda.is_available() else torch.device('cpu')

    backbones = ['resnet', 'wideresnet', 'resnext', 'densenet', 'efficientnet']
    methods = ['-dbadrp', '']

    heads = f'|{"Models":^5}|{"Loss":^5}|{"NormAcc":^5}|{"AbnormAcc":^5}|{"TotalAcc":^5}|{"Specificity":^5}|{"Sensitivity":^5}|{"AUROC":^5}|\n'
    heads += f'|-----|-----|-----|-----|-----|-----|-----|-----|'

    roc_list = list()
    cm_list = list()
    df = pandas.DataFrame()
    for k in range(5):
        test_loader = get_kfold_dataloader3d(cfg, k, cfg.dataset.validset_name)
        for backbone in backbones:
            torch.cuda.empty_cache()
            for m in methods:
                weight_files = f'{backbone}{m}{k}k_best.pt'
                cfg.backbone = f'{backbone}{m}'
                metrics = main_worker(cfg, device, test_loader, write_wrong_predict_items, weight_files)
                roc_list.append([*metrics.pop('roc')])
                cm_list.append(metrics.pop('cm'))
                metrics['model'] = weight_files
                for key, v in metrics.items():
                    metrics[key] = v.item() if isinstance(v, torch.Tensor) else v
                df = pd.concat([df, pd.DataFrame([metrics])], ignore_index=True)

    mean_cm = torch.zeros_like(cm_list[0])
    for cm in cm_list:
        mean_cm += cm
    # mean_cm = mean_cm // 5
    print(mean_cm)

    make_roc_curve_from_klist(roc_list, cfg.backbone)

    df = df.sort_values(by=['model'])
    mean_df = pd.DataFrame(columns=df.columns)
    mean_df.insert(5, 'std', None)

    for i in range(len(df) // 5):
        i = i * 5
        mean_list = df.iloc[i:i + 5][["Loss", "acc", "specificity", "sensitivity", "auroc"]].mean().tolist()
        auroc_std = df.iloc[i:i + 5][["auroc"]].std().values[0]
        model_name = df.iloc[i]['model'].split('0')[0]

        mean_list.append(auroc_std)
        mean_list.append(model_name)
        mean_df.loc[len(mean_df)] = mean_list
    print(mean_df)

    # mean_df.to_csv('inference_result_cross5.csv')
    # df.to_csv('inference_result_cross5_origin.csv')


if __name__ == '__main__':
    main()
