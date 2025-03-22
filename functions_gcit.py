# %% Necessary Packages
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics.pairwise import rbf_kernel
from scipy.stats import ks_2samp
from scipy.stats import wilcoxon
import numpy as np
import random
from scipy import stats
import time
from collections import defaultdict
import numpy as np
import warnings
from scipy.stats import rankdata


def same(x):
    return x

def cube(x):
    return np.power(x, 3)

def negexp(x):
    return np.exp(-np.abs(x))


def generate_samples_random(size=1000, sType='CI', dx=1, dy=1, dz=20, nstd=1, fixed_function='linear',
                            debug=False, normalize = True, seed = None, dist_z = 'gaussian'):
    '''Generate CI,I or NI post-nonlinear samples
    1. Z is independent Gaussian or Laplace
    2. X = f1(<a,Z> + b + noise) and Y = f2(<c,Z> + d + noise) in case of CI
    Arguments:
        size : number of samples
        sType: CI, I, or NI
        dx: Dimension of X
        dy: Dimension of Y
        dz: Dimension of Z
        nstd: noise standard deviation
        f1, f2 to be within {x,x^2,x^3,tanh x, e^{-|x|}, cos x}

    Output:
        Samples X, Y, Z
    '''
    if seed == None:
        np.random.seed()
    else:
        np.random.seed(seed)

    if fixed_function == 'linear':
        f1 = same
        f2 = same
    else:
        I1 = random.randint(2, 6)
        I2 = random.randint(2, 6)

        if I1 == 2:
            f1 = np.square
        elif I1 == 3:
            f1 = cube
        elif I1 == 4:
            f1 = np.tanh
        elif I2 == 5:
            f1 = negexp
        else:
            f1 = np.cos

        if I2 == 2:
            f2 = np.square
        elif I2 == 3:
            f2 = cube
        elif I2 == 4:
            f2 = np.tanh
        elif I2 == 5:
            f2 = negexp
        else:
            f2 = np.cos
    if debug:
        print(f1, f2)

    num = size

    if dist_z =='gaussian':
        cov = np.eye(dz)
        mu = np.ones(dz)
        Z = np.random.multivariate_normal(mu, cov, num)
        Z = np.matrix(Z)

    elif dist_z == 'laplace':
        Z = np.random.laplace(loc=0.0, scale=1.0, size=num*dz)
        Z = np.reshape(Z,(num,dz))
        Z = np.matrix(Z)

    Ax = np.random.rand(dz, dx)
    for i in range(dx):
        Ax[:, i] = Ax[:, i] / np.linalg.norm(Ax[:, i], ord=1)
    Ax = np.matrix(Ax)
    Ay = np.random.rand(dz, dy)
    for i in range(dy):
        Ay[:, i] = Ay[:, i] / np.linalg.norm(Ay[:, i], ord=1)
    Ay = np.matrix(Ay)

    Axy = np.random.rand(dx, dy)
    for i in range(dy):
        Axy[:, i] = Axy[:, i] / np.linalg.norm(Axy[:, i], ord=1)
    Axy = np.matrix(Axy)

    temp = Z * Ax
    m = np.mean(np.abs(temp))
    nstd = nstd * m 

    if sType == 'CI':
        X = f1(Z * Ax + nstd * np.random.multivariate_normal(np.zeros(dx), np.eye(dx), num))
        Y = f2(Z * Ay + nstd * np.random.multivariate_normal(np.zeros(dy), np.eye(dy), num))
    elif sType == 'I':
        X = f1(nstd * np.random.multivariate_normal(np.zeros(dx), np.eye(dx), num))
        Y = f2(nstd * np.random.multivariate_normal(np.zeros(dy), np.eye(dy), num))
    else:
        X = np.random.multivariate_normal(np.zeros(dx), np.eye(dx), num)
        Y = f2(2 * X * Axy + Z * Ay)

    if normalize == True:
        Z = (Z - Z.min()) / (Z.max() - Z.min())
        X = (X - X.min()) / (X.max() - X.min())
        Y = (Y - Y.min()) / (Y.max() - Y.min())

    return np.array(X), np.array(Y), np.array(Z)

# def pc_ks(pvals):
#     """ Compute the area under power curve and the Kolmogorov-Smirnoff
#     test statistic of the hypothesis that pvals come from the uniform
#     distribution with support (0, 1).
#     """
#     if pvals.size == 0:
#         return [-1, -1]
#     if -1 in pvals or -2 in pvals:
#         return [-1, -1]
#     pvals = np.sort(pvals)
#     cdf = ecdf(pvals)
#     auc = 0
#     for (pv1, pv2) in zip(pvals[:-1], pvals[1:]):
#         auc += integrate.quad(cdf, pv1, pv2)[0]
#     auc += integrate.quad(cdf, pvals[-1], 1)[0]
#     _, ks = kstest(pvals, 'uniform')
#     return auc, ks

# def np2r(x):
#     """ Convert a numpy array to an R matrix.
#     Args:
#         x (dim0, dim1): A 2d numpy array.
#     Returns:
#         x_r: An rpy2 object representing an R matrix isometric to x.
#     """
#     if 'rpy2' not in sys.modules:
#         raise ImportError(("rpy2 is not installed.",
#                 " Cannot convert a numpy array to an R vector."))
#     try:
#         dim0, dim1 = x.shape
#     except IndexError:
#         raise IndexError("Only 2d arrays are supported")
#     return R.r.matrix(R.FloatVector(x.flatten()), nrow=dim0, ncol=dim1)

def fdr(truth, pred, axis=None):
    """ Computes False discovery rate
    """
    return ((pred==1) & (truth==0)).sum(axis=axis) / pred.sum(axis=axis).astype(float).clip(1,np.inf)

def tpr(truth, pred, axis=None):
    """ Computes true positive rate
    """
    return ((pred==1) & (truth==1)).sum(axis=axis) / truth.sum(axis=axis).astype(float).clip(1,np.inf)

def true_positives(truth, pred, axis=None):
    """ Computes number of true positive
    """
    return ((pred==1) & (truth==1)).sum(axis=axis)

def false_positives(truth, pred, axis=None):
    """ Computes number of false positive
    """
    return ((pred==1) & (truth==0)).sum(axis=axis)

def bh(p, fdr):
    """ From vector of p-values and desired false positive rate,
    returns significant p-values with Benjamini-Hochberg correction
    """
    p_orders = np.argsort(p)
    discoveries = []
    m = float(len(p_orders))
    for k, s in enumerate(p_orders):
        if p[s] <= (k+1) / m * fdr:
            discoveries.append(s)
        else:
            break
    return np.array(discoveries, dtype=int)



def mmd_squared(X, Y, gamma = 1):

    X = X.reshape((len(X)), 1)
    Y = Y.reshape((len(Y)), 1)

    K_XX = rbf_kernel(X, gamma=gamma)
    K_YY = rbf_kernel(Y, gamma=gamma)
    K_XY = rbf_kernel(X, Y, gamma=gamma)

    n = K_XX.shape[0]
    m = K_YY.shape[0]

    mmd_squared = (np.sum(K_XX)-np.trace(K_XX))/(n*(n-1)) + (np.sum(K_YY)-np.trace(K_YY))/(m*(m-1)) - 2 * np.sum(K_XY) / (m * n)

    return mmd_squared

def correlation(X,Y):
    X = X.reshape((len(X)))
    Y = Y.reshape((len(Y)))
    return np.abs(np.corrcoef(X, Y)[0, 1])

def kolmogorov(X,Y):

    X = X.reshape((len(X)))
    Y = Y.reshape((len(Y)))
    return ks_2samp(X, Y)[0]

def wilcox(X,Y):

    X = X.reshape((len(X)))
    Y = Y.reshape((len(Y)))
    return wilcoxon(X, Y)[0]


'''
X = np.random.normal(0,2,500)
Y = np.random.normal(0,2,500)

kolmogorov(X,Y)
'''

def rdc(x, y, f=np.sin, k=20, s=1/6., n=1):
    """
    Computes the Randomized Dependence Coefficient
    x,y: numpy arrays 1-D or 2-D
         If 1-D, size (samples,)
         If 2-D, size (samples, variables)
    f:   function to use for random projection
    k:   number of random projections to use
    s:   scale parameter
    n:   number of times to compute the RDC and
         return the median (for stability)
    According to the paper, the coefficient should be relatively insensitive to
    the settings of the f, k, and s parameters.
    
    Source: https://github.com/garydoranjr/rdc
    """
    _, x_dim = x.shape
    _, y_dim = y.shape
    if not isinstance(x, np.ndarray):
        x = x.numpy()
    y = y.numpy()
    x = x.reshape((len(x)), x_dim)
    y = y.reshape((len(y)), y_dim)
    
    if n > 1:
        values = []
        for i in range(n):
            try:
                values.append(rdc(x, y, f, k, s, 1))
            except np.linalg.linalg.LinAlgError: pass
        return np.median(values)

    if len(x.shape) == 1: x = x.reshape((-1, 1))
    if len(y.shape) == 1: y = y.reshape((-1, 1))

    # Copula Transformation
    cx = np.column_stack([rankdata(xc, method='ordinal') for xc in x.T])/float(x.size)
    cy = np.column_stack([rankdata(yc, method='ordinal') for yc in y.T])/float(y.size)

    # Add a vector of ones so that w.x + b is just a dot product
    O = np.ones(cx.shape[0])
    X = np.column_stack([cx, O])
    Y = np.column_stack([cy, O])

    # Random linear projections
    Rx = (s/X.shape[1])*np.random.randn(X.shape[1], k)
    Ry = (s/Y.shape[1])*np.random.randn(Y.shape[1], k)
    X = np.dot(X, Rx)
    Y = np.dot(Y, Ry)

    # Apply non-linear function to random projections
    fX = f(X)
    fY = f(Y)

    # Compute full covariance matrix
    C = np.cov(np.hstack([fX, fY]).T)

    # Due to numerical issues, if k is too large,
    # then rank(fX) < k or rank(fY) < k, so we need
    # to find the largest k such that the eigenvalues
    # (canonical correlations) are real-valued
    k0 = k
    lb = 1
    ub = k
    while True:

        # Compute canonical correlations
        Cxx = C[:k, :k]
        Cyy = C[k0:k0+k, k0:k0+k]
        Cxy = C[:k, k0:k0+k]
        Cyx = C[k0:k0+k, :k]

        eigs = np.linalg.eigvals(np.dot(np.dot(np.linalg.pinv(Cxx), Cxy),
                                        np.dot(np.linalg.pinv(Cyy), Cyx)))

        # Binary search if k is too large
        if not (np.all(np.isreal(eigs)) and
                0 <= np.min(eigs) and
                np.max(eigs) <= 1):
            ub -= 1
            k = (ub + lb) // 2
            continue
        if lb == ub: break
        lb = k
        if ub == lb + 1:
            k = ub
        else:
            k = (ub + lb) // 2

    return np.sqrt(np.max(eigs))



# %% GCIT Function
'''
Inputs:
 - z: Confounder variables, this is the conditioning set
 - x: First target variable
 - y: Second target variable

Hyper-parameters (=Default values)
 - mu: WGAN parameter = 1
 - eta: WGAN regularization parameter = 10
 - lamda: Information network parameter = 10
 - statistic: sets the comparison function between generated and true samples (rho in the algorithm pseudocode)

'''

def GCIT(x, y, z, statistic = "rdc", lamda = 10, normalize=True, verbose=False, n_iter=1000,debug=False, test_prop=1/3):

    if normalize:
        z = (z - z.min()) / (z.max() - z.min())
        x = (x - x.min()) / (x.max() - x.min())
        y = (y - y.min()) / (y.max() - y.min())

    # %% Parameters
    # 1. # of samples
    n = len(z[:, 0])

    # define training and testing subsets, training for learning the sampler and 
    # testing for computing test statistic. Set 2/3 and 1/3 as default
    # change it to control the size in our test
    training_prop = 1-test_prop
    x_train, y_train, z_train = x[:int(n*training_prop),], y[:int(n*training_prop),], z[:int(n*training_prop),] 
    x_test, y_test, z_test = x[int(training_prop*n):,], y[int(training_prop*n):,], z[int(training_prop*n):,] 

    n = len(z_train[:, 0])
    
    # 2. # of confounders
    z_dim = len(z_train[0, :])

    # 3. # of target variables of interest (can be changed if needed)
    x_dim = len(x_train[0, :])

    # 3. # of random and hidden dimensions
    if z_dim <= 20:
        v_dim = int(3)
        h_dim = int(3)

    else:
        v_dim = int(z_dim / 10)
        h_dim = int(z_dim / 10)

    # 4. size of minibatch
    mb_size = 64

    # 5. WGAN parameters
    eta = 10
    lr = 1e-4

    # %% Necessary Functions

    # 1. Xavier Initialization Definition
    def xavier_init(size):
        in_dim = size[0]
        xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
        return tf.random.normal(shape=size, stddev=xavier_stddev)

    # 2. Sample from normal distribution: Random variable generation
    def sample_V(m, n):
        #out = np.random.rand(m, n)
        #out = np.random.laplace(loc=0.0, scale=1.0, size=n * m)
        #out = np.reshape(out, (m, n))
        out = np.random.normal(0., np.sqrt(1. / 3), size=[m, n])
        return out

    # 3. Sample from the real data (Mini-batch index sampling)
    def sample_Z(m, n):
        return np.random.permutation(m)[:n]

    # 4. Permutation for MINE computation
    def Permute(x):
        n = len(x)
        idx = np.random.permutation(n)
        out = x[idx]
        return out


    # %% Placeholders

    # 1. Target Feature
    X = tf.compat.v1.placeholder(tf.float32, shape=[None, x_dim])
    # 2. Target Permuted Feature
    X_hat = tf.compat.v1.placeholder(tf.float32, shape=[None, x_dim])
    # 3. Random Variable V
    V = tf.compat.v1.placeholder(tf.float32, shape=[None, v_dim])
    # 3. Confounder Z
    Z = tf.compat.v1.placeholder(tf.float32, shape=[None, z_dim])

    # %% Network Building

    # %% 1. WGAN Discriminator 
    # - one hidden layer as default even though more may be needed for complex data generation processes
    WD_W1 = tf.Variable(xavier_init([x_dim + z_dim, h_dim]))
    WD_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

    WD_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
    WD_b2 = tf.Variable(tf.zeros(shape=[h_dim]))

    WD_W3 = tf.Variable(xavier_init([h_dim, x_dim]))
    WD_b3 = tf.Variable(tf.zeros(shape=[x_dim]))

    theta_WD = [WD_W1, WD_W3, WD_b1, WD_b3]

    # %% 2. Generator
    # Input: Z and V
    G_W1 = tf.Variable(xavier_init([z_dim + v_dim, h_dim]))
    G_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

    G_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
    G_b2 = tf.Variable(tf.zeros(shape=[h_dim]))

    G_W3 = tf.Variable(xavier_init([h_dim, x_dim]))
    G_b3 = tf.Variable(tf.zeros(shape=[x_dim]))

    theta_G = [G_W1, G_W3, G_b1, G_b3]

    # %% 3. MINE
    # Input: X and tilde X
    # For X
    M_W1A = tf.Variable(xavier_init([x_dim]))
    M_W1B = tf.Variable(xavier_init([x_dim]))
    M_b1 = tf.Variable(tf.zeros(shape=[x_dim]))

    # For tilde X
    M_W2A = tf.Variable(xavier_init([x_dim]))
    M_W2B = tf.Variable(xavier_init([x_dim]))
    M_b2 = tf.Variable(tf.zeros(shape=[x_dim]))

    # Combine
    M_W3 = tf.Variable(xavier_init([x_dim]))
    M_b3 = tf.Variable(tf.zeros(shape=[x_dim]))

    theta_M = [M_W1A, M_W1B, M_W2A, M_W2B, M_W3, M_b1, M_b2, M_b3]

    # %% Functions
    # 1. Generator
    def generator(z, v):
        inputs = tf.concat(axis=1, values=[z, v])
        G_h1 = tf.nn.tanh(tf.matmul(inputs, G_W1) + G_b1)
        #G_h2 = tf.nn.tanh(tf.matmul(G_h1, G_W2) + G_b2)
        #G_out = tf.nn.sigmoid(tf.matmul(G_h2, G_W3) + G_b3)

        G_out = tf.nn.sigmoid(tf.matmul(G_h1, G_W3) + G_b3)

        return G_out

    # 2. WGAN Discriminator
    def WGAN_discriminator(x, z):
        inputs = tf.concat(axis=1, values=[x, z])
        WD_h1 = tf.nn.relu(tf.matmul(inputs, WD_W1) + WD_b1)
        #WD_h2 = tf.nn.relu(tf.matmul(WD_h1, WD_W2) + WD_b2)
        #WD_out = (tf.matmul(WD_h2, WD_W3) + WD_b3)

        WD_out = (tf.matmul(WD_h1, WD_W3) + WD_b3)

        return WD_out

    # 3. MINE
    def MINE(x, x_hat):
        M_h1 = tf.nn.tanh(M_W1A * x + M_W1B * x_hat + M_b1)
        M_h2 = tf.nn.tanh(M_W2A * x + M_W2B * x_hat + M_b2)
        M_out = (M_W3 * (M_h1 + M_h2) + M_b3)

        Exp_M_out = tf.exp(M_out)

        return M_out, Exp_M_out

    # %% Combination across the networks
    # 1. Generator
    G_sample = generator(Z, V)

    # 2. WGAN Outputs for real and fake
    WD_real = WGAN_discriminator(X, Z)
    WD_fake = WGAN_discriminator(G_sample, Z)

    # 3. MINE Computation
    # Without permutation
    M_out, _ = MINE(X, G_sample)
    # With permutation
    _, Exp_M_out = MINE(X_hat, G_sample)

    # 4. WGAN Loss Replacement of Clipping algorithm to Penalty term
    eps = tf.random.uniform([mb_size, 1], minval=0., maxval=1.)
    X_inter = eps * X + (1. - eps) * G_sample

    grad = tf.gradients(WGAN_discriminator(X_inter, Z), [X_inter])[0]
    grad_norm = tf.sqrt(tf.reduce_sum((grad) ** 2 + 1e-8, axis=1))
    grad_pen = eta * tf.reduce_mean((grad_norm - 1) ** 2)

    # %% Loss function
    # 1. WGAN Loss, aim to make WD_fake small and WD_real big
    WD_loss = tf.reduce_mean(WD_fake) - tf.reduce_mean(WD_real) + grad_pen

    # 2. MINE Loss
    M_loss = lamda * (tf.reduce_sum(tf.reduce_mean(M_out, axis=0) - \
                                    tf.math.log(tf.reduce_mean(Exp_M_out, axis=0))))

    # 3. Generator loss, aim make WD_fake high
    G_loss = -tf.reduce_mean(WD_fake) + lamda * M_loss

    # Solver
    WD_solver = (tf.compat.v1.train.AdamOptimizer(learning_rate=lr, beta1=0.5).minimize(WD_loss, \
                                                                                        var_list=theta_WD))
    G_solver = (tf.compat.v1.train.AdamOptimizer(learning_rate=lr, beta1=0.5).minimize(G_loss, var_list=theta_G))
    M_solver = (tf.compat.v1.train.AdamOptimizer(learning_rate=lr, beta1=0.5).minimize(-M_loss, var_list=theta_M))

    # %% Sessions
    sess = tf.compat.v1.Session()
    sess.run(tf.global_variables_initializer())

    Generator_loss = []
    Mine_loss = []
    WDiscriminator_loss = []

    # %% Iterations
    for it in range(n_iter):

        for _ in range(5):
            # %% WGAN and MINE Training

            # Random variable generation
            V_mb = sample_V(mb_size, v_dim)

            # Minibatch sampling
            Z_idx = sample_Z(n, mb_size)
            Z_mb = z_train[Z_idx, :]
            X_mb = x_train[Z_idx]
            X_perm_mb = Permute(X_mb)

            # 1. WGAN Training
            _, WD_loss_curr = sess.run([WD_solver, WD_loss], \
                                       feed_dict={X: X_mb, Z: Z_mb, V: V_mb, X_hat: X_perm_mb})
            # 2. MINE Training
            _, M_loss_curr = sess.run([M_solver, M_loss], \
                                      feed_dict={X: X_mb, Z: Z_mb, V: V_mb, X_hat: X_perm_mb})

        # Random variable generation
        V_mb = sample_V(mb_size, v_dim)

        # Minibatch sampling
        Z_idx = sample_Z(n, mb_size)
        Z_mb = z_train[Z_idx, :]
        X_mb = x_train[Z_idx]
        X_perm_mb = Permute(X_mb)

        # Generator training
        _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={X: X_mb, Z: Z_mb, V: V_mb, X_hat: X_perm_mb})

        Generator_loss.append(G_loss_curr)
        WDiscriminator_loss.append(WD_loss_curr)
        Mine_loss.append(M_loss_curr)

        # %% Intermediate Losses
        if verbose and it % 499 == 0:
            print('Iter: {}'.format(it))
            print('Generator_loss: {:.4}'.format(G_loss_curr))
            print('WD_loss: {:.4}'.format(WD_loss_curr))
            print('M_loss: {:.4}'.format(M_loss_curr))
            print()
            
        # stop training if discriminator loss is sufficiently low (heuristic)
        if abs(WD_loss_curr) < 0.1:
            break
            
    if verbose:
        # plot training losses
        plt.plot(range(n_iter), WDiscriminator_loss, range(n_iter), Generator_loss, range(n_iter), Mine_loss)
        plt.legend(('WGAN Discriminator', 'Generator', 'MINE'),
                   loc='upper right')
        plt.title('Training losses')
        plt.tight_layout()
        plt.show()

    # %% Compute test statistic
    # 1. Number of samples for null computation
    n_samples = 1000
    rho = []

    # 2. Choice of statistic
    if statistic == "corr":
        stat = correlation
    if statistic == "mmd":
        stat = mmd_squared
    if statistic == "kolmogorov":
        stat = kolmogorov
    if statistic == "wilcox":
        stat = wilcox
    if statistic == "rdc":
        stat = rdc
   
    # 3. Generate and collect samples on testing data
    for sample in range(n_samples):
        x_hat = sess.run([G_sample], feed_dict={Z: z_test, V: sample_V(len(z_test[:, 0]), v_dim)})[0]
        rho.append(stat(x_hat, y_test))

    # 4. p-value computation as a two-sided test
    p_value = min(sum(rho < stat(x_test.reshape(len(x_test), x_dim), y_test)) / n_samples,
                  sum(rho > stat(x_test.reshape(len(x_test), x_dim), y_test)) / n_samples)

    
    if debug:
        print('Statistics of x_hat ', stats.describe(x_hat))
        print('Statistics of x_train ',stats.describe(x_test))
        print('Statistics of generated rho ', stats.describe(rho))
        print('Observed rho',stat(x_test.reshape(len(x_test)), y_test))

    return(p_value)
