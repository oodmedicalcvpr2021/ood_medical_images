import torch
import torch.nn as nn
import torch.nn.functional as F


def get_loss(logits, confidence, target, args):
    if args.mode == 'baseline':
        task_loss = nn.CrossEntropyLoss()(logits, target.cuda())
        total_loss = task_loss
        aux_loss = 0.0
        return total_loss, task_loss, aux_loss

    if args.mode == 'oe':
        task_loss = F.cross_entropy(logits[:len(target)], target)
        oe_loss = -(logits[len(target):].mean(1) - torch.logsumexp(logits[len(target):], dim=1)).mean()
        total_loss = task_loss + args.args.beta * oe_loss
        return total_loss, task_loss, oe_loss

    elif args.mode == 'devries':
        task_probs = F.softmax(logits, dim=-1)
        confidence_prob = torch.sigmoid(confidence)

        _, num_classes = logits.size()
        one_hot_target = nn.functional.one_hot(target, num_classes=num_classes)

        # Make sure we don't have any numerical instability
        eps = 1e-12
        task_probs = torch.clamp(task_probs, 0. + eps, 1. - eps)
        confidence_prob = torch.clamp(confidence_prob, 0. + eps, 1. - eps)

        if args.use_hint:
            b = (torch.rand_like(confidence_prob) < args.hint_rate).float()
            conf = confidence_prob * b + (1 - b)
        else:
            conf = confidence_prob

        pred_new = task_probs * conf.expand_as(task_probs) + one_hot_target * (1 - conf.expand_as(one_hot_target))
        pred_new = torch.log(pred_new)
        xentropy_loss = F.nll_loss(pred_new, target)

        confidence_loss = torch.mean(-torch.log(confidence_prob))
        total_loss = xentropy_loss + (args.lmbda * confidence_loss)
        if args.use_budget:
            if args.beta > confidence_loss.data:
                args.lmbda = args.lmbda / 1.01
            elif args.beta <= confidence_loss.data:
                args.lmbda = args.lmbda / 0.99

        return total_loss, xentropy_loss, confidence_loss

def get_confidence(logits, confidence, args):

    if args.mode == 'baseline':
        probability = torch.softmax(logits, dim=-1)
        pred_value, _ = torch.max(probability.data, 1)
        out = pred_value.cpu().numpy()
    elif args.mode == 'devries':
        out = torch.sigmoid(confidence).data.cpu().numpy()
    elif args.mode == 'oe' or args.mode == 'energy':
        probability = torch.softmax(logits, dim=-1)
        pred_value, _ = torch.max(probability.data, 1)
        out = pred_value.cpu().numpy()
    else:
        raise NotImplementedError
    return out

