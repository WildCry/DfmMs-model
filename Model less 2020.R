library(kimfilter)
library(data.table)
library(maxLik)
library(gridExtra)
library(fredr)
library(purrr)
library(tidyverse)

# Fetch Dataset ############################################################################################################################################################################################################################################################

fredr_set_key("d59606a150e09c54fd5158bac863da0d")  # Use your actual FRED API key

# Function to fetch data for a given series ID
fetch_series_data <- function(series_id) {
  fredr_series_observations(series_id = series_id) %>%
    mutate(series_id = series_id) %>%
    select(date, value, series_id)
}

# List of series IDs
series_ids <- c("IPMAN", "W875RX1", "CMRMTSPL", "PAYEMS")

# Fetch and combine data from all series
data <- map_dfr(series_ids, fetch_series_data)

# Convert the data to wide format
data <- data %>%
  pivot_wider(names_from = series_id, values_from = value)

data = na.omit(data)

summary(data)

# Plotting
ggplot(data, aes(x = date)) + 
  geom_line(aes(y = IPMAN, color = "IPMAN")) + 
  geom_line(aes(y = W875RX1, color = "W875RX1")) +
  geom_line(aes(y = CMRMTSPL, color = "CMRMTSPL")) +
  geom_line(aes(y = PAYEMS, color = "PAYEMS")) +
  theme_minimal() +
  labs(title = "Time Series of Various Economic Indicators",
       x = "Date",
       y = "Indicator Value",
       color = "Series") +
  scale_color_manual(values = c("IPMAN" = "blue", "W875RX1" = "red", "CMRMTSPL" = "green", "PAYEMS" = "purple")) +
  theme(legend.position = "bottom")

# data preprocessing  ########################################################################################################################################################################################################################################################################################

# Ensure data is a data.table
setDT(data)

#Log the data
data.log = copy(data)
data.log[, c(series_ids) := lapply(.SD, log), .SDcols = c(series_ids)]

#Difference the data
data.logd = copy(data.log)
data.logd[, c(series_ids) := lapply(.SD, function(x){
  x - shift(x, type = "lag", n = 1)
}), .SDcols = c(series_ids)]

#Center the data
data.logds = copy(data.logd)
data.logds[, c(series_ids) := lapply(.SD, scale, scale = FALSE), .SDcols = c(series_ids)]

#Transpose the data
yt = t(data.logds[, c(series_ids), with = F])



# Build Model ################################################################################################################################################################################################################################################################################################

#State space model for the Stock and Watson Markov Switching Dynamic Common Factor model
msdcf_ssm = function(par, yt, n_states = NULL){
  #Get the number of states
  n_states = length(unique(unlist(lapply(strsplit(names(par)[grepl("p_", names(par))], "p_"), function(x){substr(x[2], 1, 1)}))))
  
  #Get the parameters
  vars = dimnames(yt)[which(unlist(lapply(dimnames(yt), function(x){!is.null(x)})))][[1]]
  phi = par[grepl("phi", names(par))]
  names(phi) = gsub("phi", "", names(phi))
  gamma = par[grepl("gamma_", names(par))]
  names(gamma) = gsub("gamma_", "", names(gamma))
  psi = par[grepl("psi_", names(par))]
  names(psi) = gsub("psi_", "", names(psi))
  sig = par[grepl("sigma_", names(par))]
  names(sig) = gsub("sigma_", "", names(sig))
  mu = par[grepl("mu", names(par))]
  names(mu) = gsub("mu_", "", names(mu))
  pr = par[grepl("p_", names(par))]
  names(pr) = gsub("p_", "", names(pr))
  states = sort(unique(substr(names(pr), 1, 1)))
  
  #Steady state probabilities
  Pm = matrix(NA, nrow = n_states, ncol = n_states)
  rownames(Pm) = colnames(Pm) = unique(unlist(lapply(names(pr), function(x){strsplit(x, "")[[1]][2]})))
  for(j in names(pr)){
    Pm[strsplit(j, "")[[1]][2], strsplit(j, "")[[1]][1]] = pr[j]
  }
  for(j in 1:ncol(Pm)){
    Pm[which(is.na(Pm[, j])), j] = 1 - sum(Pm[, j], na.rm = TRUE)
  }
  
  #Build the transition matrix
  phi_dim = max(c(length(phi)), length(unique(sapply(strsplit(names(gamma), "\\."), function(x){x[2]}))))
  psi_dim = sapply(unique(sapply(strsplit(names(psi), "\\."), function(x){x[1]})), function(x){
    max(as.numeric(sapply(strsplit(names(psi)[grepl(paste0("^", x), names(psi))], "\\."), function(x){x[2]})))
  })
  Fm = matrix(0, nrow = phi_dim + length(psi), ncol = phi_dim + length(psi), 
              dimnames = list(
                c(paste0("ct.", 0:(phi_dim - 1)), 
                  unlist(lapply(names(psi_dim), function(x){paste0("e_", x, ".", 0:(psi_dim[x] - 1))}))),
                c(paste0("ct.", 1:phi_dim), 
                  unlist(lapply(names(psi_dim), function(x){paste0("e_", x, ".", 1:psi_dim[x])})))
              ))
  Fm["ct.0", paste0("ct.", names(phi))] = phi
  for(i in 1:length(vars)){
    Fm[paste0("e_", i, ".0"), 
       paste0("e_", names(psi)[grepl(paste0("^", i), names(psi))])] = psi[grepl(paste0("^", i), names(psi))]
  }
  diag(Fm[intersect(rownames(Fm), colnames(Fm)), intersect(rownames(Fm), colnames(Fm))]) = 1
  Fm = array(Fm, dim = c(nrow(Fm), ncol(Fm), n_states), dimnames = list(rownames(Fm), colnames(Fm), states))
  
  #Build the observation matrix
  Hm = matrix(0, nrow = nrow(yt), ncol = nrow(Fm), dimnames = list(rownames(yt), rownames(Fm)))
  for(i in 1:length(vars)){
    Hm[i, paste0("ct.", sapply(strsplit(names(gamma)[grepl(paste0("^", i), names(gamma))], "\\."), function(x){x[2]}))] = 
      gamma[grepl(paste0("^", i), names(gamma))]
  }
  diag(Hm[, paste0("e_", 1:length(vars), ".0")]) = 1
  Hm = array(Hm, dim = c(nrow(Hm), ncol(Hm), n_states), dimnames = list(rownames(Hm), colnames(Hm), states))
  
  #Build the state covariance matrix
  #Set the dynamic common factor standard deviation to 1
  Qm = matrix(0, ncol = ncol(Fm), nrow = nrow(Fm), dimnames = list(rownames(Fm), rownames(Fm)))
  Qm["ct.0", "ct.0"] = 1
  for(i in 1:length(vars)){
    Qm[paste0("e_", i, ".0"), paste0("e_", i, ".0")] = sig[names(sig) == i]^2
  } 
  Qm = array(Qm, dim = c(nrow(Qm), ncol(Qm), n_states), dimnames = list(rownames(Qm), colnames(Qm), states))
  
  #Build the observation equation covariance matrix
  Rm = matrix(0, ncol = nrow(Hm), nrow = nrow(Hm), dimnames = list(vars, vars))
  Rm = array(Rm, dim = c(nrow(Rm), ncol(Rm), n_states), dimnames = list(rownames(Rm), colnames(Rm), states))
  
  #State intercept matrix: the Markov switching mean matrix
  Dm = matrix(0, nrow = nrow(Fm), ncol = 1, dimnames = list(rownames(Fm), NULL))
  Dm = array(Dm, dim = c(nrow(Dm), 1, n_states), dimnames = list(rownames(Fm), NULL, states))
  for(i in names(mu)){
    Dm["ct.0", , i] = mu[i]
  }
  
  #Observation equation intercept matrix
  Am = matrix(0, nrow = nrow(Hm), ncol = 1)
  Am = array(Am, dim = c(nrow(Am), ncol(Am), n_states), dimnames = list(vars, NULL, states))
  
  #Initialize the filter for each state
  B0 = matrix(0, nrow(Fm), 1)
  P0 = diag(nrow(Fm))
  B0 = array(B0, dim = c(nrow(B0), ncol(B0), n_states), dimnames = list(rownames(Fm), NULL, states))
  P0 = array(P0, dim = c(nrow(P0), ncol(P0), n_states), dimnames = list(rownames(B0), colnames(B0), states))
  for(i in states){
    B0[,,i] = solve(diag(ncol(Fm)) - Fm[,,i]) %*% Dm[,,i]
    VecP_ll = solve(diag(prod(dim(Fm[,,i]))) - kronecker(Fm[,,i], Fm[,,i])) %*% matrix(as.vector(Qm[,,i]), ncol = 1)
    P0[,,i] = matrix(VecP_ll[, 1], ncol = ncol(Fm))
  }
  
  return(list(B0 = B0, P0 = P0, Am = Am, Dm = Dm, Hm = Hm, Fm = Fm, Qm = Qm, Rm = Rm, Pm = Pm))
}





# The rest ########################################################################################################################################################################################################################################################################################

#Set the initial values
init = c(phi1 = 0.8760, phi2 = -0.2171, 
         mu_u = 0.2802, mu_d = -1.5700,
         p_dd = 0.8406, p_uu = 0.9696,
         psi_1.1 = 0.0364, psi_1.2 = -0.0008,
         psi_2.1 = -0.2965, psi_2.2 = -0.0657,
         psi_3.1 = -0.3959, psi_3.2 = -0.1903,
         psi_4.1 = -0.2436, psi_4.2 = 0.1281,
         gamma_1.0 = 0.0058, gamma_1.1 = -0.0033,
         gamma_2.0 = 0.0011,  
         gamma_3.0 = 0.0051, gamma_3.1 = -0.0033 , 
         gamma_4.0 = 0.0012, gamma_4.1 = -0.0005, 
         gamma_4.2 = 0.0001, gamma_4.3 = 0.0002,
         sigma_1 = 0.0048, sigma_2 = 0.0057, 
         sigma_3 = 0.0078, sigma_4 = 0.0013)

#Set the constraints
ineqA = matrix(0, nrow = 20, ncol = length(init), dimnames = list(NULL, names(init)))
#Stationarity constraints
ineqA[c(1, 2), c("phi1", "phi2")] = rbind(c(1, 1), c(-1, -1))
ineqA[c(3, 4), grepl("psi_1", colnames(ineqA))] = rbind(c(1, 1), c(-1, -1))
ineqA[c(5, 6), grepl("psi_2", colnames(ineqA))] = rbind(c(1, 1), c(-1, -1))
ineqA[c(7, 8), grepl("psi_3", colnames(ineqA))] = rbind(c(1, 1), c(-1, -1))
ineqA[c(9, 10), grepl("psi_4", colnames(ineqA))] = rbind(c(1, 1), c(-1, -1))
#Non-negativity constraints
diag(ineqA[c(11, 12, 13, 14), grepl("sigma_", colnames(ineqA))]) = 1
ineqA[c(15, 16), "p_dd"] = c(1, -1)
ineqA[c(17, 18), "p_uu"] = c(1, -1)
#Up/down states must be positive/negative
ineqA[19, "mu_u"] = 1
ineqA[20, "mu_d"] = -1
ineqB = matrix(c(rep(1, 10), 
                 rep(0, 4), 
                 c(0, 1), 
                 c(0, 1), 
                 rep(0, 2)), nrow = nrow(ineqA), ncol = 1)

#Define the objective function
objective = function(par, yt){
  ssm = msdcf_ssm(par, yt)
  return(kim_filter(ssm, yt, smooth = FALSE)$lnl)
}

#Solve the model
solve = maxLik(logLik = objective, start = init, method = "BFGS", 
               finalHessian = FALSE, hess = NULL, 
               control = list(printLevel = 2, iterlim = 10000), 
               constraints = list(ineqA = ineqA, ineqB = ineqB), 
               yt = yt)

#Get the estimated state space model
ssm = msdcf_ssm(solve$estimate, yt)

#Get the column means and standard deviations
M = matrix(unlist(data.logd[, lapply(.SD, mean, na.rm = TRUE), .SDcols = c(vars)]), 
           ncol = 1, dimnames = list(vars, NULL))

#Get the steady state coefficient matrices
Pm = matrix(ss_prob(ssm[["Pm"]]), ncol = 1, dimnames = list(rownames(ssm[["Pm"]]), NULL))
Hm = Reduce("+", lapply(dimnames(ssm[["Hm"]])[[3]], function(x){
  Pm[x, ]*ssm[["Hm"]][,, x]
}))
Fm = Reduce("+", lapply(dimnames(ssm[["Fm"]])[[3]], function(x){
  Pm[x, ]*ssm[["Fm"]][,, x]
}))

#Final K_t is approximation to steady state K
filter = kim_filter(ssm, yt, smooth = TRUE)
K = filter$K_t[,, dim(filter$K_t)[3]]
W = solve(diag(nrow(K)) - (diag(nrow(K)) - K %*% Hm) %*% Fm) %*% K
d = (W %*% M)[1, 1]

#Get the intercept terms
gamma = Hm[, grepl("ct", colnames(Hm))]
D = M - gamma %*% matrix(rep(d, ncol(gamma)))

#Initialize first element of the dynamic common factor
Y1 = t(data.log[, c(vars), with = F][1, ])
initC = function(par){
  return(sum((Y1 - D - gamma %*% par)^2))
}
C10 = optim(par = Y1, fn = initC, method = "BFGS", control = list(trace = FALSE))$par[1]
Ctt = rep(C10, ncol(yt))

#Build the rest of the level of the dynamic common factor
ctt = filter$B_tt[which(rownames(Fm) == "ct.0"), ]
for(j in 2:length(Ctt)){
  Ctt[j] = ctt[j] + Ctt[j - 1] + c(d)
}
Ctt = data.table(date = data$date, dcf = Ctt, d.dcf = ctt)
prob = data.table(date = data$date, data.table(filter$Pr_tt))
colnames(prob) = c("date", paste0("pr_", dimnames(ssm$Dm)[[3]]))
uc = merge(Ctt, prob, by = "date", all = TRUE)

#Plot the outputs
g1 = ggplot(melt(data.log, id.vars = "date")[, "value" := scale(value), by = "variable"]) + 
  ggtitle("Data", subtitle = "Log Levels (Rescaled)") + 
  scale_y_continuous(name = "Value") + 
  scale_x_date(name = "") + 
  geom_line(aes(x = date, y = value, group = variable, color = variable)) + 
  theme_minimal() + theme(legend.position = "bottom") + guides(color = guide_legend(title = NULL))

g2 = ggplot( melt(data.logds, id.vars = "date")) + 
  ggtitle("Data", subtitle = "Log Differenced & Standardized") + 
  scale_y_continuous(name = "Value") + 
  scale_x_date(name = "") + 
  geom_hline(yintercept = 0, color = "black") + 
  geom_line(aes(x = date, y = value, group = variable, color = variable)) + 
  theme_minimal() + theme(legend.position = "bottom") + guides(color = guide_legend(title = NULL))

toplot3 = melt(uc, id.vars = "date")
d_range1 = range(toplot3[variable == "dcf", ]$value, na.rm = TRUE)
p_range1 = range(toplot3[variable %in% colnames(uc)[grepl("pr_", colnames(uc))], ]$value, na.rm = TRUE)
toplot3[variable %in% colnames(uc)[grepl("pr_", colnames(uc))], "value" := (value - p_range1[1])/diff(p_range1) * diff(d_range1) + d_range1[1], by = "variable"]
g3 = ggplot() +  
  ggtitle("Dynamic Common Factor", subtitle = "Levels") + 
  scale_x_date(name = "") +
  geom_hline(yintercept = 0, color = "grey") + 
  geom_line(data = toplot3[variable == "dcf", ], 
            aes(x = date, y = value, group = variable, color = variable)) + 
  theme_minimal() + theme(legend.position = "bottom") + 
  guides(color = guide_legend(title = NULL), fill = guide_legend(title = NULL)) + 
  scale_color_manual(values = "black") + 
  scale_y_continuous(name = "Value", limits = range(toplot3[variable == "dcf", ]$value, na.rm = TRUE), 
                     sec.axis = sec_axis(name = "Probability", ~((. - d_range1[1])/diff(d_range1) * diff(p_range1) + p_range1[1]) * 100)) + 
  geom_ribbon(data = toplot3[variable %in% "pr_d", ], 
              aes(x = date, ymin = d_range1[1], ymax = value, group = variable, fill = variable), alpha = 0.5) + 
  scale_fill_manual(values = c("red", "green"))

toplot4 = melt(uc, id.vars = "date")
d_range2 = range(toplot4[variable %in% c("d.dcf"), ]$value, na.rm = TRUE)
p_range2 = range(toplot4[variable %in% colnames(uc)[grepl("pr_", colnames(uc))], ]$value, na.rm = TRUE)
toplot4[variable %in% colnames(uc)[grepl("pr_", colnames(uc))], "value" := (value - p_range2[1])/diff(p_range2) * diff(d_range2) + d_range2[1], by = "variable"]
g4 = ggplot() +  
  ggtitle("Dynamic Common Factor", subtitle = "Differenced") + 
  scale_x_date(name = "") +
  geom_hline(yintercept = 0, color = "grey") + 
  geom_line(data = toplot4[variable %in% c("d.dcf"), ], 
            aes(x = date, y = value, group = variable, color = variable)) + 
  theme_minimal() + theme(legend.position = "bottom") + 
  guides(color = guide_legend(title = NULL), fill = guide_legend(title = NULL)) + 
  scale_color_manual(values = "black") + 
  scale_y_continuous(name = "Value", limits = range(toplot4[variable %in% c("d.dcf"), ]$value, na.rm = TRUE), 
                     sec.axis = sec_axis(name = "Probability", ~((. - d_range2[1])/diff(d_range2) * diff(p_range2) + p_range2[1]) * 100)) + 
  geom_ribbon(data = toplot4[variable %in% "pr_d", ], 
              aes(x = date, ymin = d_range2[1], ymax = value, group = variable, fill = variable), alpha = 0.5) + 
  scale_fill_manual(values = c("red", "green"))

grid.arrange(g1, g2, g3, g4, layout_matrix = matrix(c(1, 3, 2, 4), nrow = 2))