import numpy as np
from evaluate import test
from numba_backward import conv2d_backward_core

class Module:
    def __init__(self, *args):
        self.args = args
        self.output_layer = False
        self.transforming_layer = False
        self.output_shape = None

    def forward(self):
        "forward wird in jeder subclass überschrieben"
        pass 

    def __call__(self, x):
        return self.forward(x)

class Parameters:
    def __init__(self, n_in, n_out, required_grad=True, K=None):
        #Alle wichtigen Parameter in einem Objekt
        self.n_in, self.n_out = n_in, n_out
        self.weight = np.zeros((n_out, n_in))
        if K is not None: self.weight = np.random.randn(n_out, n_in, K, K) * np.sqrt(2 / (n_in * K * K ))
        self.bias = np.zeros(n_out)
        self.z = np.zeros_like(self.bias)
        self.grad = np.zeros_like(self.weight)
        self.bias_grad = np.zeros_like(self.bias)
        self.required_grad = required_grad

    def init(self, init_method='he'):

        #Initalisierungsmethode auswählen
        if init_method == 'he':

            #Zwei Arrays anhand der richtigen Input/Output Dimensionen
            W = np.random.randn(self.n_out, self.n_in) * np.sqrt(2/self.n_in)
            b = np.random.rand(self.n_out) 

        self.weight = W
        self.bias = b

class Linear(Module, Parameters):
    def __init__(self, *args):
        input, output = args

        #Layer bekommt Parameter übergeben
        Module.__init__(self, *args)
        Parameters.__init__(self, input, output)

        #Loss wenn output, inputs wenn input
        self.inputs = None

        self.init() #Standardinitialisierung

    def forward(self, x):
        x = np.array(x)

        #Input speichern für Input Layer im Backprop
        self.inputs = x
        self.z = (self.weight @ x) + self.bias
        return self.z
    
    def parameters(self):
        #Zur Visualisierung
        return [self.weight, self.bias, self.grad, self.required_grad]
    
    def backward(self, delta, crit=None, y=None):

        # Wenn Output Linear
        if self.output_layer:
            delta = crit.derivitave(*y)

        # Gradienten
        self.grad = np.outer(delta, self.inputs) #Outerprodukt *
        self.bias_grad = delta.copy()

        # Delta für vorherige Schicht (dL/da_{l-1})
        delta_prev = self.weight.T @ delta

        return delta_prev

class Activation(Module):
    def __init__(self):
        super().__init__()
        self.z = None

    def forward(self, x):
        #Wird von allen Aktivierungsfunktionen überschrieben
        pass
    
    def derivitave(self, x):
        #Wird von allen Aktivierungsfunktionen überschrieben
        pass

    def backward(self, delta, crit=None, y=None):

        if self.transforming_layer:
            delta = delta.flatten()

        # Spezialfall CrossEntropyLoss + Softmax
        if self.output_layer and isinstance(crit, CrossEntropy) and isinstance(self, Softmax):
            return crit.derivitave(*y)
        
        #Ableitung der Aktivierungsfunktion
        act_derived = self.derivitave(self.z)

        # Wenn output ist delta ableitung vom loss
        if self.output_layer:
            delta = crit.derivitave(*y) 

        if isinstance(self, Softmax):
            return act_derived @ delta
        else: 
            return delta * act_derived
    
class ReLU(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = np.array(x)

        #Relu Funktion
        act = np.maximum(0, x)
        self.z = act

        # Wenn Conv2d davor dannach ist
        if self.transforming_layer:
            act = act.reshape(self.output_shape)
            #self.z = self.z.reshape(self.output_shape)

        return act
    
    def derivitave(self, x):
        #Ableitung der Relu
        x = np.array(x)
        return (x > 0).astype(float)

class Softmax(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        #Exponenzieren der Eingabe
        x = np.array(x - np.max(x))
        sum = np.sum(np.exp(x))
        self.z = np.exp(x) / sum
        return self.z

    def derivitave(self, x):
        x = np.array(x)
        self.z = np.diag(x) - np.outer(x, x.T)
        return self.z

class Dropout(Activation):
    def __init__(self, p=0.6):
        super().__init__()
        self.mask = None
        self.p = p

    def forward(self, x):
        x = np.array(x)
        #Wenn dropout am output ist einfach als statisches layer weitergeben
        if self.output_layer:
            return x

        #Filter anwenden
        self.mask = (np.random.rand(*x.shape) > self.p).astype(float)
        self.z = (x * self.mask) / (1 - self.p)
        return self.z
    
    def derivitave(self, x):
        return self.mask / (1 - self.p)

class Criterion:
    def __init__(self, criterion):
        self.criterion = criterion
        self.y_pred = None
        self.y_true = None

    def logits(self, y_true):
        y_vec = np.zeros_like(self.y_pred)
        y_vec[int(y_true)] = 1.0
        self.y_true = y_vec

    def backward(self, y_pred, y_true):

        output, modules = y_pred
        self.y_pred, self.y_true = output, y_true

        #One Hot Encoding der y_true für CrEnt
        if isinstance(self.criterion, CrossEntropy): self.logits(self.y_true)

        # Für die Outputschicht delta
        delta = modules[-1].backward(
            delta=None,
            crit=self.criterion,
            y=(self.y_pred, self.y_true)
        )

        # Für alle anderen Schichten
        for layer in reversed(modules[:-1]):
            delta = layer.backward(delta=delta)

class MSE(Criterion):
    def __init__(self, y_pred=None, y_true=None):
        super().__init__(self)

    def value(self):
        # mse value
        diff = self.y_pred - self.y_true
        return np.mean(diff ** 2)
    
    def derivitave(self, y_pred, y_true):
        # Ableitung von MSE
        diff = y_pred - y_true
        return 2 * diff 

class CrossEntropy(Criterion):
    def __init__(self):
        super().__init__(self)
        self.eps = 1e-12

    def value(self):
        y_pred = np.clip(self.y_pred, self.eps, 1-self.eps)
        return np.sum(self.y_true * np.log(y_pred))

    def derivitave(self, y_pred, y_true):
        y_pred = np.clip(y_pred, self.eps, 1-self.eps)
        return y_pred - y_true 

class Optim:

    #Optimiere der LossFunction
    def __init__(self):
        pass
    
    class SDG:
        def __init__(self, parameters, lr):

            #Bekommt das ganze Netz ausser die Aktivierungsfunktionen
            self.modules = [m for m in parameters if isinstance(m, (Linear, Conv2d))]
            self.lr = lr # Lernrate

        def step(self):
            #Gradietenschritt -> weights werden mit grad überschrieben
            for module in self.modules:
                module.weight -= self.lr * module.grad
                module.bias -= self.lr * module.bias_grad

    class Adam:
        def __init__(self, parameters, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
            self.modules = [m for m in parameters if isinstance(m, (Linear, Conv2d))]

            self.lr = lr
            self.beta1, self.beta2 = betas
            self.eps = eps

            # Schrittzähler für Bias-Korrektur
            self.t = 0

            # Moment-Schätzer für Gewichte und Bias
            self.m_w = {m: np.zeros_like(m.weight) for m in self.modules}
            self.v_w = {m: np.zeros_like(m.weight) for m in self.modules}
            self.m_b = {m: np.zeros_like(m.bias) for m in self.modules}
            self.v_b = {m: np.zeros_like(m.bias) for m in self.modules}

        def step(self):
            if not self.modules:
                return

            self.t += 1

            for module in self.modules:
                g_w = module.grad
                g_b = module.bias_grad

                # Falls keine Gradienten gesetzt wurden, Layer überspringen
                if g_w is None or g_b is None:
                    continue

                # 1. Moment (m) aktualisieren
                mw = self.m_w[module] = self.beta1 * self.m_w[module] + (1.0 - self.beta1) * g_w
                mb = self.m_b[module] = self.beta1 * self.m_b[module] + (1.0 - self.beta1) * g_b

                # 2. Moment (v) aktualisieren
                vw = self.v_w[module] = self.beta2 * self.v_w[module] + (1.0 - self.beta2) * (g_w ** 2)
                vb = self.v_b[module] = self.beta2 * self.v_b[module] + (1.0 - self.beta2) * (g_b ** 2)

                # Bias-Korrektur
                mw_hat = mw / (1.0 - self.beta1 ** self.t)
                mb_hat = mb / (1.0 - self.beta1 ** self.t)
                vw_hat = vw / (1.0 - self.beta2 ** self.t)
                vb_hat = vb / (1.0 - self.beta2 ** self.t)

                # Parameter-Update
                module.weight -= self.lr * mw_hat / (np.sqrt(vw_hat) + self.eps)
                module.bias   -= self.lr * mb_hat / (np.sqrt(vb_hat) + self.eps)

class Sequential(Module):

    #Frame für das Netz, bekommt alle Modules
    def __init__(self, *args):
        super().__init__(*args)
        self.modules = args

        for i in range(len(self.modules) - 1):       # Flattening layer fest
            layer = self.modules[i]
            next_layer = self.modules[i + 1]
            if isinstance(layer, Conv2d) and not isinstance(next_layer, Conv2d):
                layer.flattening_layer = True
            if not isinstance(layer, Conv2d) and isinstance(next_layer, Conv2d):
                layer.transforming_layer = True

        #Output festlegen
        self.modules[-1].output_layer = True

    def forward(self, x):

        #Forwarden jedes Moduls und ergebnis zurückgeben
        output = np.array(x)
        prev_dim = self.modules[0].output_shape
        for module in self.modules: 
            if module.transforming_layer:
                module.output_shape = prev_dim
            output = module.forward(output)
            prev_dim = module.output_shape

        #Zeiger auf alle Module in dem Netz hier zurück für Backprop
        return output, self.modules
    
    def add_module(self, module : Module):
        pass

    def parameters(self):
        return self.modules
    
    def fit(self, X_train, y_train, X_test, y_test, criteron, optimizer, epochs=50, verbose=True, classification=True, testing=False):
        
        training_losses = []
        for epoch in range(epochs):
            loss = 0
            for X, y_true in zip(X_train, y_train):

                #Forward prediction
                y_pred = self.forward(X)

                # Backward-Gradienten anpassung
                criteron.backward(y_pred, y_true)

                # Gradientenabstieg
                optimizer.step()

                # Loss der Epoche berechnen und speichern
                loss += criteron.value()    
            training_losses.append((loss/len(X_train)))

            test_acc, test_prec = 0, 0
            if testing:
                test_acc, test_prec = test(self, X_test, y_test, criteron, classification)

            if verbose: print(f"Epoche: {epoch+1} Trainings-Loss: {loss / len(X_train):.4f} Test-Accuracy: {test_acc:.4f} ")
        return training_losses

    def evaluate(self, X_test, y_test, criterion, classification):
        test(self, X_test, y_test, criterion, classification, visualize=True)

class Conv2d(Module, Parameters):
    def __init__(self, n_in, n_out, K):
        Parameters.__init__(self, n_in, n_out, K=K)
        Module.__init__(self)
        # n_out wie viele Filter/maps die nächste schicht hat, n_in tiefe des Eingangs
        self.z = None
        self.input = None

        # Übergang zum Linear
        self.flattening_layer = False

    def forward(self, X):
        X = np.array(X)
        self.input = X

        H, W, Cx = X.shape
        n_out, n_in, K, _ = self.weight.shape
        H_out = H - K + 1
        W_out = W - K + 1

        patches = np.zeros((H_out * W_out, Cx * K * K), dtype=X.dtype)
        idx = 0
        for i in range(H_out):
            for j in range(W_out):
                patch = X[i:i+K, j:j+K, :]             # (K, K, Cx)
                patches[idx] = patch.transpose(2, 0, 1).reshape(-1)
                idx += 1

        kernels_flat = self.weight.reshape(n_out, -1)

        out_flat = patches @ kernels_flat.T

        self.z = out_flat.reshape(H_out, W_out, n_out)
        self.output_shape = self.z.shape

        if self.flattening_layer:
            return self.flatten(self.z)
        return self.z

    def convolute(self, X, K):
        H, W, Cx = X.shape
        Ck, _, K = K.shape
        assert Cx == Ck

        #Output Dimensionen 
        H_out = H - K + 1
        W_out = W - K + 1

        filter_output = np.zeros((H_out, W_out))

        #Blocks raussuchen und summe über multiplikation mit kernel berechnen (Faltung)
        for h in range(H_out):
            row = []
            for w in range(W_out):
                # Block von momentanen Kernel-Scanning
                block = X[ h : h+K , w : w+K , : ]
                block = np.transpose(block, (2, 0, 1))
                conv = np.sum(block * K)
                filter_output[h, w] = conv

        return filter_output

    def backward(self, delta=None, crit=None, y=None):
        # Falls Flattening aktiviert war: Delta zurück in Feature-Map-Form bringen
        if self.flattening_layer:
            delta = delta.reshape(self.output_shape)

        # Falls dieser Conv-Layer die Output-Schicht ist, delta aus dem Kriterium holen
        if (self.output_layer or delta is None) and crit is not None and y is not None:
            y_pred, y_true = y
            delta_scalar = crit.derivitave(y_pred, y_true)
            delta = np.full(self.z.shape, delta_scalar)

        X = self.input  # (H, W, C_in)

        # Kernberechnung an Numba-Funktion auslagern
        delta_prev, grad = conv2d_backward_core(delta, X, self.weight)

        # Gradienten im Objekt speichern
        self.grad = grad

        return delta_prev

    def flatten(self, feature_map):
        return feature_map.reshape(-1)




#early stopping
#layernorm
#batchnorm
#netzwerk ersteller
    

