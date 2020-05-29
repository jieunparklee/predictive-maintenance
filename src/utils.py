import numpy as np

def compute_likelihood (x, mean, cov) : # multivariate gaussian distribution # log likelihood
    k = len(mean)
    m_dist_x = np.dot((x-mean).transpose(),np.linalg.inv(cov))
    m_dist_x = np.dot(m_dist_x, (x-mean)) # Mahalanobis distance
    return -1/2 * (np.log(np.linalg.det(cov)) + m_dist_x + (len(mean) * np.log(2*np.pi)))

def forecast (X, model, n_past, n_future) : 
    num_features = X.shape[2]
    full_pred = []
    for t in range(len(X)-n_past+1) :
        pred = []
        history = X[t:t+n_past-1].reshape(1, n_past-1, num_features) # (# samples, time_steps, # features)
        out = X[t+n_past-1].reshape(1, 1, num_features)
        for f in range(n_future) :
            X_pred = np.append(history, out, axis=1)
            history = X_pred[:,1:,:]
            out = model.predict(X_pred)[-1].reshape(1, 1, num_features)
            pred.append(out.reshape(num_features))
        full_pred.append(pred)
    return np.array(full_pred)

