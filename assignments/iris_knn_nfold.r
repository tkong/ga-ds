library(class)
library(ggplot2)

#################################################
# Manual parameters and data load
#################################################

data <- iris                # create copy of iris dataframe

set.seed(1)         # initialize random seed for consistency
max.folds <- 10
max.k <- 10

#################################################
# Functions
#################################################

knn.wrapper <- function(testPartitionNum, k, dframe, labelcol) {
  labels <- dframe[, labelcol]    
  
  train.index <- which(dframe$partition.num != testPartitionNum)
  test.index <- which(dframe$partition.num == testPartitionNum)

  train.labels <- dframe[train.index, labelcol]      # extract training set labels
  test.labels <- dframe[test.index, labelcol]        # extract test set labels
    
  dframe[, labelcol] <- NULL                         # remove labels from feature set
  
  train.data <- dframe[train.index,]                 # perform train/test split
  test.data <- dframe[test.index,]                   # note use of neg index...different than Python!

  knn.fit <- knn(train = train.data,  # training set, not parition x
                 test = test.data,    # test set, partition x
                 cl = train.labels,   # true labels
                 k = k)               # number of NN to poll

  this.err <- sum(test.labels != knn.fit) / length(test.labels)    # gzn err
  
  cat('\n', 'Parameters: Test Partition = ', testPartitionNum, ', k = ', k, sep='')     # print params
  cat('\n', 'Error = ', this.err, sep='')     # print params
  print(table(test.labels, knn.fit))          # print confusion matrix   
  
  return(this.err)
}

knn.nfold <- function(n, k, dframe, labelcol) {
  # create n-fold partition of dataset
  # perform knn classification n times
  # n-fold generalization error = average over all iterations

  sampleSize <- nrow(data)     # total number of rows
  
  dframe <- dframe[sample(sampleSize),] # shuffle data
  dframe$partition.num <- rep(1:n, rep(ceiling(sampleSize/n), n))[1:sampleSize] # assign partitions; remainder is distributed to partitions ascending 
   
  avg.error <- mean(sapply(1:n, knn.wrapper, k=k, dframe=dframe, labelcol=labelcol))
  
  cat('\n AVERAGE GENERALIZATION ERROR (', k, ' nearest neighbor,', n, ' fold cross validation) = ', avg.error, '\n', sep='')
  return(avg.error)
}


#################################################
# Run knn with n-fold cross-validation for every combination of n and k
#################################################

nk <- expand.grid(n=2:max.folds, k=1:max.k) # data frame containing every combination of 2:max.folds and 1:max.k

results <- data.frame()       # initialize results object
results <- data.frame(n.folds = as.factor(nk$n), k = nk$k, avg.gzn.error = mapply(knn.nfold, nk$n, nk$k, dframe=list(data), labelcol="Species"))
print(results)

#################################################
# Plot average generalization error against k for every n-folds series 
#################################################

results.plot <- ggplot(results, aes(x=k, y=avg.gzn.error, color=n.folds)) 
results.plot <- results.plot + geom_point() + geom_line() + theme_bw() + ggtitle('knn n-fold cross validation') + xlab('k') + ylab('Average Generalization Error')
print(results.plot)
