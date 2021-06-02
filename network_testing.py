import torch


def test_net(net, criterion, test_loader, device):
    # тестирование
    test_loss = 0
    correct = 0
    need_resize = net.need_resize

    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            if need_resize:
                data = data.view(-1, 28 * 28)

            net_out = net(data)
            # Суммируем потери со всех партий
            test_loss += criterion(net_out, labels).data
            pred = net_out.data.max(1)[1]  # получаем индекс максимального значения
            # сравниваем с целевыми данными, если совпадает добавляем в correct
            correct += pred.eq(labels.data).sum()

    test_loss /= len(test_loader.dataset)
    test_acc = float(100. * correct / len(test_loader.dataset))
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        test_acc))
    return test_acc
