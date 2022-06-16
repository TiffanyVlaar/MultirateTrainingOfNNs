# Code accompanying "Multirate Training of Neural Networks", ICML 2022

We present the torch.optimizer corresponding to the following multirate algorithm (this is Algorithm 1 in the ICML paper). <br>

The base algorithm used is SGD with momentum.

To generate the transfer learning results in Section 4 one simply loads in the optimizer using:

    from Optimizer_multirate import Multirate
    optimizer = Multirate(net.parameters(),lr=h,momentum=mu)  # where net is the considered neural network, and h and mu are as defined in Algorithm 1

Then one initializes the momenta in epoch 0 for the first batch using: optimizer.initmom() (after calling loss.backward). <br>
We set the fast parameters to be the fully connected (fc) layer. <br>
Inside the training loop:
    
    for batch_idx, (inputs, targets) in enumerate(loader_train): 
        if (batch_idx+1) % k == 0: #where k is as defined in Algorithm 1
            for param in net.parameters():
                param.requires_grad = True

            for param in net.fc.parameters():
                param.requires_grad = False

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets) 
            loss.backward()

            optimizer.stepslow()

            for param in net.parameters():
                param.requires_grad = False

            for param in net.fc.parameters():
                param.requires_grad = True

        else:
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets) 
            loss.backward()

            optimizer.stepfast()
