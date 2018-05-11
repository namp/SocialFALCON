/* This file is part of SocialFALCON matrix factorization algorithms code contribution

SocialFALCON is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

SocialFALCON is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

See <http://www.gnu.org/licenses/>

Authors: N. Ampazis and T. Emmanouilidis (2018)
Original codebase contribution by George Tsagas

Revision:1

*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdarg.h>
#include <mysql.h>

#define PREDICTION_MODE 0
#define MIN_EPOCHS       10                  // Minimum number of epochs per feature
#define MIN_IMPROVEMENT  0.00005              // Minimum improvement required to continue current feature

#define INIT_SEED_Mb       -0.3             // sqrtf(GLOBAL_AVERAGE/(float)TOTAL_FEATURES)   // Initialization value for features
#define INIT_VARIANCE_Mb   0.20             // variance range from the INIT_SEED value
#define INIT_Mb            (INIT_SEED_Mb + (2.0*(rand()/(float)(RAND_MAX)) - 1.0)*INIT_VARIANCE_Mb) // INIT + rand[-INIT_VARIANCE, +INIT_VARIANCE] 

#define INIT_SEED_Cb       0.0              // sqrtf(GLOBAL_AVERAGE/(float)TOTAL_FEATURES)   // Initialization value for features
#define INIT_VARIANCE_Cb   0.010            // variance range from the INIT_SEED value
#define INIT_Cb            (INIT_SEED_Cb + (2.0*(rand()/(float)(RAND_MAX)) - 1.0)*INIT_VARIANCE_Cb) // INIT + rand[-INIT_VARIANCE, +INIT_VARIANCE] 

#define INIT_SEED_M        0.0               // sqrtf(GLOBAL_AVERAGE/(float)TOTAL_FEATURES)   // Initialization value for features
#define INIT_VARIANCE_M    0.001             // variance range from the INIT_SEED value
#define INIT_M             (INIT_SEED_M + (2.0*(rand()/(float)(RAND_MAX)) - 1.0)*INIT_VARIANCE_M) // INIT + rand[-INIT_VARIANCE, +INIT_VARIANCE] 

#define INIT_SEED_C        0.0               // sqrtf(GLOBAL_AVERAGE/(float)TOTAL_FEATURES)   // Initialization value for features
#define INIT_VARIANCE_C    0.001             // variance range from the INIT_SEED value
#define INIT_C             (INIT_SEED_C + (2.0*(rand()/(float)(RAND_MAX)) - 1.0)*INIT_VARIANCE_C) // INIT + rand[-INIT_VARIANCE, +INIT_VARIANCE] 

double LRATE1u  =            0.0001;        // Learning rate parameter for features
double LAMDA1u  =            0.1;        // reg for features
double LRATE1m  =            0.0001;        // Learning rate parameter for features
double LAMDA1m  =            0.1;        // reg for features

double LRATE2ub =            0.0001;        // Learning rate parameter for biases
double LAMDA2ub =            0.1;        // reg for biases
double LRATE2mb =            0.0001;        // Learning rate parameter for biases
double LAMDA2mb =            0.1;          // reg for biases

double LAMDAT =              0.1;          // reg for biases


struct connection_details
{
    char *server;
    char *user;
    char *password;
    char *database;
};
 
MYSQL* mysql_connection_setup(struct connection_details mysql_details)
{
     // first of all create a mysql instance and initialize the variables within
    MYSQL *connection = mysql_init(NULL);
 
    // connect to the database with the details attached.
    if (!mysql_real_connect(connection,mysql_details.server, mysql_details.user, mysql_details.password, mysql_details.database, 0, NULL, 0)) {
      printf("Conection error : %s\n", mysql_error(connection));
      exit(1);
    }
    return connection;
}
 
MYSQL_RES* mysql_perform_query(MYSQL *connection, char *sql_query)
{
   // send the query to the database
   if (mysql_query(connection, sql_query))
   {
      printf("MySQL query error : %s\n", mysql_error(connection));
      exit(1);
   }
 
   return mysql_use_result(connection);
}


double randn(void);

void array_min(double a[], double *min, int *position, int MAX_EPOCHS);

void double_array_fill_zeros (double *my_array, unsigned int size_of_my_array);

double predict_svd_rating (int movieId, int custId, int TOTAL_FEATURES);

int rnd(int max);


void create_txt_file ();
void write_txt_file (unsigned int customer_id, unsigned short movie_id);
void close_txt_file ();


void calc_features(int TOTAL_FEATURES, int MAX_EPOCHS);

double sigmoid (double alpha);


FILE *lgfile=NULL;
void lg(char *fmt,...);
void lgopen(int argc, char**argv);
void error(char *fmt,...);

double final_probe_rmse=0.0;
int final_epochs_for_probe;


//////////////////////////database connection///////////////////////////////////////
  char query_string[200];

  MYSQL *conn;    // the connection
  MYSQL_RES *res; // the results
  MYSQL_ROW row;  // the results row (line by line)
 
  struct connection_details mysqlD;

int TOTAL_MOVIES;
int TOTAL_CUSTOMERS;
int TOTAL_RATES;
int TOTAL_PROBES;  
double GLOBAL_AVERAGE;
double GLOBAL_SCALED_AVERAGE;
int min_r,max_r, rating_range;
double avg;


// ****** SVD *********** //
double **movie_features;     // Array of features by movie (using doubles to save space)
double **cust_features;   // Array of features by customer (using doubles to save space)
double *m_bias;
double *c_bias;

int **user_movies;
int *user_movies_size;
int **user_ratings;

double **movie_features_gradients;     // Array of gradients for features by movie (using doubles to save space)
double **cust_features_gradients;   // Array of gradients for features by customer (using doubles to save space)
double *m_bias_gradients;
double *c_bias_gradients;
int *user_connections_size;
int *user_reverse_connections_size;
int **user_connections;
int **user_reverse_connections;


// ******************** //


// *** PROBE ***//
int *probe_customers;
int *probe_movies;
int *probe_real_scores;
// *** //

char algorithm_name[20];
float MIN_ERROR = 1000.0;


main (int argc, char**argv) {

  lgopen(argc,argv);

  double prediction;
  unsigned int i,h;
  time_t start, stop;
  double diff;
  int TOTAL_FEATURES = atoi(argv[5]);
  int MAX_EPOCHS = atoi(argv[6]);

   /* start timer */
  start = time(NULL);  

  mysqlD.server = argv[1];  // where the mysql database is
  mysqlD.user = argv[2];   // the root user of mysql 
  mysqlD.password = argv[3]; // the password of the root user in mysql
  mysqlD.database = argv[4]; // the databse to pick
 
  // connect to the mysql database
  conn = mysql_connection_setup(mysqlD);


sprintf(query_string,"SELECT count(DISTINCT item_id) FROM item_mapping");

res = mysql_perform_query(conn,query_string);


while ((row = mysql_fetch_row(res)) !=NULL) {
         TOTAL_MOVIES=atoi(row[0]);
}

//clean up the database result set
mysql_free_result(res);



sprintf(query_string,"SELECT count(DISTINCT user_id) FROM user_mapping");

res = mysql_perform_query(conn,query_string);


while ((row = mysql_fetch_row(res)) !=NULL) {
         TOTAL_CUSTOMERS=atoi(row[0]);
}

//clean up the database result set
mysql_free_result(res);

sprintf(query_string,"SELECT count(*) FROM train");

res = mysql_perform_query(conn,query_string);

while ((row = mysql_fetch_row(res)) !=NULL) {
         TOTAL_RATES=atoi(row[0]);
}

//clean up the database result set
mysql_free_result(res);

sprintf(query_string,"SELECT count(*) FROM probe");

res = mysql_perform_query(conn,query_string);



while ((row = mysql_fetch_row(res)) !=NULL) {
         TOTAL_PROBES=atoi(row[0]);
}

//clean up the database result set
mysql_free_result(res);


sprintf(query_string,"SELECT avg(rating_value) FROM train");

res = mysql_perform_query(conn,query_string);


while ((row = mysql_fetch_row(res)) !=NULL) {
         GLOBAL_AVERAGE=atof(row[0]);
}

//clean up the database result set
mysql_free_result(res);



// Get maximum and minimum ratings from the ratings table

sprintf(query_string,"SELECT MAX(rating_value) FROM ratings");

res = mysql_perform_query(conn,query_string);

while ((row = mysql_fetch_row(res)) !=NULL) {
         max_r=atoi(row[0]);
}

 /* clean up the database result set */
mysql_free_result(res);


sprintf(query_string,"SELECT MIN(rating_value) FROM ratings");

res = mysql_perform_query(conn,query_string);

while ((row = mysql_fetch_row(res)) !=NULL) {
         min_r=atoi(row[0]);
}

 /* clean up the database result set */
mysql_free_result(res);

rating_range=max_r-min_r;


avg = (GLOBAL_AVERAGE - min_r) / rating_range;
GLOBAL_SCALED_AVERAGE = log(avg / (1 - avg));

movie_features = ( double** )malloc(TOTAL_MOVIES * sizeof(double *));

  if(movie_features == NULL)
    {
    fprintf(stderr, "out of memory for movie features array\n");
    exit(-1);
        }
  for(i = 0; i < TOTAL_MOVIES; i++)
    {
    movie_features[i] = ( double* )malloc(TOTAL_FEATURES * sizeof(double));
    if(movie_features[i] == NULL)
      {
      fprintf(stderr, "out of memory for movie features row\n");
      exit(-1);
      }
    }


cust_features = ( double** )malloc(TOTAL_CUSTOMERS * sizeof(double *));

  if(cust_features == NULL)
    {
    fprintf(stderr, "out of memory for customer features array\n");
    exit(-1);
        }
  for(i = 0; i < TOTAL_CUSTOMERS; i++)
    {
    cust_features[i] = ( double* )malloc(TOTAL_FEATURES * sizeof(double));
    if(cust_features[i] == NULL)
      {
      fprintf(stderr, "out of memory for customer features row\n");
      exit(-1);
      }
    }

m_bias =  (double *)malloc(sizeof(double)*TOTAL_MOVIES);

c_bias =  (double *)malloc(sizeof(double)*TOTAL_CUSTOMERS);


movie_features_gradients = ( double** )malloc(TOTAL_MOVIES * sizeof(double *));

  if(movie_features_gradients == NULL)
    {
    fprintf(stderr, "out of memory for movie gradients array\n");
    exit(-1);
        }
  for(i = 0; i < TOTAL_MOVIES; i++)
    {
    movie_features_gradients[i] = ( double* )malloc(TOTAL_FEATURES * sizeof(double));
    if(movie_features_gradients[i] == NULL)
      {
      fprintf(stderr, "out of memory for movie gradients array row\n");
      exit(-1);
      }
    }


cust_features_gradients = ( double** )malloc(TOTAL_CUSTOMERS * sizeof(double *));

  if(cust_features_gradients == NULL)
    {
    fprintf(stderr, "out of memory for customer gradients array\n");
    exit(-1);
        }
  for(i = 0; i < TOTAL_CUSTOMERS; i++)
    {
    cust_features_gradients[i] = ( double* )malloc(TOTAL_FEATURES * sizeof(double));
    if(cust_features_gradients[i] == NULL)
      {
      fprintf(stderr, "out of memory for customer gradients row\n");
      exit(-1);
      }
    }


m_bias_gradients =  (double *)malloc(sizeof(double)*TOTAL_MOVIES);

c_bias_gradients =  (double *)malloc(sizeof(double)*TOTAL_CUSTOMERS);


user_connections = ( int** )malloc(TOTAL_CUSTOMERS * sizeof(int *));

  if(user_connections == NULL)
    {
    fprintf(stderr, "out of memory for user connections\n");
    exit(-1);
    }
  
user_connections_size =  (int *)malloc(sizeof(int)*TOTAL_CUSTOMERS);


user_reverse_connections = ( int** )malloc(TOTAL_CUSTOMERS * sizeof(int *));

  if(user_reverse_connections == NULL)
    {
    fprintf(stderr, "out of memory for user reverse connections\n");
    exit(-1);
   }


user_reverse_connections_size =  (int *)malloc(sizeof(int)*TOTAL_CUSTOMERS);


// ***************** //

user_movies = ( int** )malloc(TOTAL_CUSTOMERS * sizeof(int *));


  if(user_movies == NULL)
    {
    fprintf(stderr, "out of memory for user connections\n");
    exit(-1);
    }


user_movies_size =  (int *)malloc(sizeof(int)*TOTAL_CUSTOMERS);

user_ratings = ( int** )malloc(TOTAL_CUSTOMERS * sizeof(int *));


   if(user_ratings == NULL)
    {
    fprintf(stderr, "out of memory for user connections\n");
    exit(-1);
    }

/* stop timer and display time */
  stop = time(NULL);
  diff = difftime(stop, start);

// *** CREATE PROBE *** //

/* start timer */
start = time(NULL);  

probe_customers = (int *)malloc(sizeof(int)*TOTAL_PROBES);
probe_movies = (int *)malloc(sizeof(int)*TOTAL_PROBES);
probe_real_scores = (int *)malloc(sizeof(int)*TOTAL_PROBES);

sprintf(query_string,"select user_id,item_id,rating_value FROM probe");

res = mysql_perform_query(conn,query_string);

 h=0;////just a counter
  while ((row = mysql_fetch_row(res)) !=NULL) {
      probe_customers[h]=atoi(row[0]);
      probe_movies[h]=atoi(row[1]);
      probe_real_scores[h]=atoi(row[2]);
      h++;
}

// ******************** //
  

  /* stop timer and display time */
  stop = time(NULL);
  diff = difftime(stop, start);

  // start timer
  start = time(NULL);  

  // RUN SVD

  sscanf(argv[0], "./%s", algorithm_name);
  lg("%s\t\t",algorithm_name);
  calc_features(TOTAL_FEATURES, MAX_EPOCHS);

  /* stop timer and display time */
  stop = time(NULL);
  diff = difftime(stop, start);
  lg("%f sec\n", diff);

exit(-1);

  // *** SAVE FEATURES ***

  // stop timer and display time 
  stop = time(NULL);
  diff = difftime(stop, start);

  exit(0);
}


//****** SVD *********


void calc_features(int TOTAL_FEATURES, int MAX_EPOCHS) {

  time_t start, stop, start_e, stop_e;
  double diff;
  double avg_diff = 0.0;
  int c, d, h,f, e, i, j, custId,vcustId, vmcustId,  cnt = 0;

  int v,w;  

  int *movie_id;
  int *rating;
  int *mutual_neighbours_id;

  int num_movies;
  int num_neighbours;
  int num_mutual_neighbours;
  int num_mutual_num_neighbours;

 
  double err, err2, p, sq, rmse_last, rmse = 2.0, probe_rmse=9998, probe_rmse_last=9999, probe_sq;
       
  int movieId;
  double cf, mf, cf_bias, mf_bias;

  int  wcustId;

  double Tuv,Tvu,Tvw;

  double cf_bias_v,cf_bias_vt,cf_bias_w,cf_bias_vu;
  double diff_uv[TOTAL_FEATURES];
  double diffUV[TOTAL_FEATURES];
  double diff_vw[TOTAL_FEATURES];
  double diffVU[TOTAL_FEATURES];
  double diffVW[TOTAL_FEATURES];
  double epochs_probe_error[MAX_EPOCHS];


  // INIT all feature values 
  for (f=0; f<TOTAL_FEATURES; f++) {
    for (i=0; i<TOTAL_MOVIES; i++) {
      movie_features[i][f] = INIT_M;
    }
    for (i=0; i<TOTAL_CUSTOMERS; i++) {
      cust_features[i][f] = INIT_C;
    }
  }


  // *** INIT biases
  for (i=0; i<TOTAL_MOVIES; i++) {
    m_bias[i] = INIT_Mb;
  }
  for (i=0; i<TOTAL_CUSTOMERS; i++) {
    c_bias[i] = INIT_Cb;
  }




////////////////First we count how many movies exist in our train dataset and store them

sprintf(query_string,"SELECT COUNT(DISTINCT item_id) FROM train");

res = mysql_perform_query(conn,query_string);

int num_train_movies;

while ((row = mysql_fetch_row(res)) !=NULL) {
         num_train_movies=atoi(row[0]);
}

/* clean up the database result set */
mysql_free_result(res);

//////////Now we select train  movies and store them in an array

int *train_movies_id;

train_movies_id = (int *)malloc(sizeof(int)*num_train_movies);

///The select query
sprintf(query_string,"SELECT DISTINCT item_id FROM train ORDER BY item_id");

res = mysql_perform_query(conn,query_string);

///fetch all selected rows

 h=0;////just a counter
  while ((row = mysql_fetch_row(res)) !=NULL) {
      train_movies_id[h]=atoi(row[0]);
      h++;
 }

/* clean up the database result set */
mysql_free_result(res);



////////////////First we count how many users exist in our train dataset and store them

sprintf(query_string,"SELECT COUNT(DISTINCT user_id) FROM train");

res = mysql_perform_query(conn,query_string);

int num_train_users;

while ((row = mysql_fetch_row(res)) !=NULL) {
         num_train_users=atoi(row[0]);
}

/* clean up the database result set */
mysql_free_result(res);


//////////Now we select all train users and store them in an array

int *train_users_id;

train_users_id = (int *)malloc(sizeof(int)*num_train_users);

///The select query
sprintf(query_string,"SELECT DISTINCT user_id FROM train ORDER BY user_id");

res = mysql_perform_query(conn,query_string);

///fetch all selected rows

 h=0;////just a counter
  while ((row = mysql_fetch_row(res)) !=NULL) {
      train_users_id[h]=atoi(row[0]);
      h++;
}

/* clean up the database result set */
mysql_free_result(res);


////////Now we have the train set users stored


//////Now we need to find all connections and reverse connection of all users

 for (c=0; c<TOTAL_CUSTOMERS;c++) {


custId = c+1;


//////Find out how many movies the user have rated

sprintf(query_string,"select count(item_id) FROM train WHERE user_id=%d",custId);

res = mysql_perform_query(conn,query_string);

while ((row = mysql_fetch_row(res)) !=NULL) {
         num_movies=atoi(row[0]);
}

 /* clean up the database result set */
mysql_free_result(res);


 if (num_movies!=0) {

user_movies_size[c] = num_movies;


user_movies[c] = ( int* )malloc(num_movies * sizeof(int));

    if(user_movies[c] == NULL)
      {
      fprintf(stderr, "out of memory for connections of customer %d\n", custId);
      exit(-1);
      }

user_ratings[c] = ( int* )malloc(num_movies * sizeof(int));

    if(user_ratings[c] == NULL)
      {
      fprintf(stderr, "out of memory for connections of customer %d\n", custId);
      exit(-1);
      }


/////select and store the movies and ratings the user have rated

sprintf(query_string,"select item_id, rating_value FROM train WHERE user_id=%d",custId);

res = mysql_perform_query(conn,query_string);

 h=0;////just a counter
  while ((row = mysql_fetch_row(res)) !=NULL) {
      user_movies[c][h]=atoi(row[0]);
      user_ratings[c][h]=atoi(row[1]);
      h++;
}

/* clean up the database result set */
mysql_free_result(res);

}

sprintf(query_string,"select count(target_user_id) from trust where source_user_id=%d",custId);

res = mysql_perform_query(conn,query_string);

///fetch all selected rows

while ((row = mysql_fetch_row(res)) !=NULL) {
         num_neighbours=atoi(row[0]);
}

 /* clean up the database result set */
mysql_free_result(res);

user_connections_size[c]=num_neighbours;

    if (num_neighbours!=0) {

user_connections[c] = ( int* )malloc(num_neighbours * sizeof(int));
    
    if(user_connections[c] == NULL)
      {
      fprintf(stderr, "out of memory for connections of customer %d\n", custId);
      exit(-1);
      }
   


sprintf(query_string,"select target_user_id from trust where source_user_id=%d",custId);

res = mysql_perform_query(conn,query_string);

///fetch all selected rows


  h=0; //Just a counter
  while ((row = mysql_fetch_row(res)) !=NULL) {
       user_connections[c][h]=atoi(row[0]);
       h++;
      }
  
 /* clean up the database result set */
mysql_free_result(res);


///////We got all neigbours for this user
} // if 


//////Now we need to find WHO FOLLOWS this user 

sprintf(query_string,"select count(source_user_id) from trust where target_user_id=%d",custId);


res = mysql_perform_query(conn,query_string);

///fetch all selected rows

while ((row = mysql_fetch_row(res)) !=NULL) {
         num_mutual_neighbours=atoi(row[0]);
}

 /* clean up the database result set */
mysql_free_result(res);

user_reverse_connections_size[c]=num_mutual_neighbours;

		if (num_mutual_neighbours!=0) {

user_reverse_connections[c] = ( int* )malloc(num_mutual_neighbours * sizeof(int));

    if(user_reverse_connections[c] == NULL)
      {
      fprintf(stderr, "out of memory for reverse connections of customer %d\n", custId);
      exit(-1);
      }
   

sprintf(query_string,"select source_user_id from trust where target_user_id=%d",custId);

res = mysql_perform_query(conn,query_string);

///fetch all selected rows

 h=0;////just a counter
  while ((row = mysql_fetch_row(res)) !=NULL) {
      user_reverse_connections[c][h]=atoi(row[0]);
      h++;
}


 /* clean up the database result set */
mysql_free_result(res);

} //if

///////We got all MUTUAL neigbours for this user


} // for (all users)


  // Keep looping until you have stopped making significant (probe_rmse) progress
    for (e=0; e < MAX_EPOCHS; e++) {
        
    start = time(NULL);
    start_e = time(NULL);
    
    cnt++;
    sq = 0;
    probe_sq = 0;
    rmse_last = rmse;
    probe_rmse_last = probe_rmse;

// RESET all feature gradients 
  for (f=0; f<TOTAL_FEATURES; f++) {
    for (i=0; i<TOTAL_MOVIES; i++) {
      movie_features_gradients[i][f] = 0.0;
    }
    for (i=0; i<TOTAL_CUSTOMERS; i++) {
      cust_features_gradients[i][f] = 0.0;
    }
  }


// *** RESET biases gradients
  for (i=0; i<TOTAL_MOVIES; i++) {
    m_bias_gradients[i] = 0.0;
  }
  for (i=0; i<TOTAL_CUSTOMERS; i++) {
    c_bias_gradients[i] = 0.0;
  }



///////continue with the train iteration

     for (c=0; c < num_train_users; c++)  {
      d=c;
      custId = train_users_id[c];

/// READY FOR ITERATIONS ////

      if (user_movies_size[custId-1]!=0) {


      for (i=0; i< user_movies_size[custId-1]; i++) {

        movieId=user_movies[custId-1][i];
      
        p = predict_svd_rating (movieId, custId, TOTAL_FEATURES);

        err2 =  ( (double)user_ratings[custId-1][i] - (rating_range*p + min_r) );

        sq += err2*err2;


        //*** train biases
        cf_bias = c_bias[custId - 1];
        mf_bias = m_bias[movieId - 1];
        c_bias_gradients[custId - 1] += (LRATE2ub * (err2 *  rating_range * p  * (1.0 - p) - LAMDA1u*LAMDA2ub * cf_bias));

        m_bias_gradients[movieId - 1] += (LRATE2mb * (err2 * rating_range * p  * (1.0 - p) - LAMDA1m*LAMDA2mb * mf_bias));

  

        for (f=0; f<TOTAL_FEATURES; f++) {      

          // Cache off old feature values
          cf = cust_features[custId - 1][f];
          mf = movie_features[movieId - 1][f];
         
         cust_features_gradients[custId - 1][f] += (LRATE1u * (err2 * rating_range * p * (1.0 - p) * mf - LAMDA1u * cf));
         movie_features_gradients[movieId - 1][f] += (LRATE1m * (err2 * rating_range * p * (1.0 - p) * cf - LAMDA1m * mf));
        }
      }

  }

} //Pass over TRAIN users

     for (c=0; c < TOTAL_CUSTOMERS; c++)  {

      d=c;
      custId = c+1;
    

cf_bias_vt=0.0;

for (f=0;f<TOTAL_FEATURES;f++) {
      diff_uv[f] = 0.0;
      }


cf_bias_vu = 0.0;


for (f=0; f<TOTAL_FEATURES; f++) {
diffVU[f] = 0.0;
}


Tuv=0.0;


//////Now we need to find all neighbours of this user

         if (user_connections_size[c]!=0) {



Tuv=1.0/user_connections_size[c];

  for (i=0;i<user_connections_size[c];i++) {

       vcustId=user_connections[c][i];
     
      for (f=0;f<TOTAL_FEATURES;f++) {
      diff_uv[f] += cust_features[vcustId - 1][f];
      }
  

     cf_bias_vt += c_bias[vcustId - 1];    

   }

///////We got all DIFFS for this user (and its neighbors)


//////Now we need to find all MUTUAL neighbours of this user 


           if (user_reverse_connections_size[c]!=0) {

////initialise variables


//////Now we need to find ALL neighbours of MUTUAL NEIGHBOURS !!

  for (i=0;i<user_reverse_connections_size[c];i++) {

     vmcustId = user_reverse_connections[c][i];

Tvu=1.0/user_connections_size[vmcustId - 1];
Tvw=Tvu;

/* Initialize variables */

 for (f=0; f<TOTAL_FEATURES; f++) {
 diff_vw[f]=0.0;
 }

 cf_bias_w=0.0;

   for (j=0; j<user_connections_size[vmcustId - 1];j++) {

      wcustId = user_connections[vmcustId - 1][j];
      
      for (f=0; f<TOTAL_FEATURES; f++) {
         diff_vw[f] += cust_features[wcustId - 1][f];
       }

       cf_bias_w += c_bias[wcustId - 1];

   }


///////We got all neigbours for this MUTUAL neighbor    


            for (f=0; f<TOTAL_FEATURES; f++) {

            diffVW[f]= cust_features[vmcustId - 1][f] - diff_vw[f]*Tvw;

            }

            cf_bias_w = c_bias[vmcustId - 1] - cf_bias_w*Tvw;


       for (f=0; f<TOTAL_FEATURES; f++) {

       diffVU[f] += Tvu * diffVW[f];

      }


       cf_bias_vu += Tvu * cf_bias_w;

}

} else {

}
} else {

}

//SOCIAL GRADIENT UPDATES

        cf_bias_v = c_bias[custId - 1] - cf_bias_vt*Tuv;
        c_bias_gradients[custId - 1] -= LRATE2ub * LAMDAT * (cf_bias_v - cf_bias_vu);

        for (f=0; f<TOTAL_FEATURES; f++) {      
         diffUV[f]= cust_features[custId - 1][f] - diff_uv[f]*Tuv;
         cust_features_gradients[custId - 1][f] -= LRATE1u * LAMDAT * (diffUV[f] - diffVU[f]);
      }
      
    if ((d!=0) && (d%10000000 == 0)){

    stop = time(NULL);
    diff = difftime(stop,start);
    printf("Done %d Customers in %f secs\n",d,diff);
    start = time(NULL);

   }
  


} // TOTAL Customers


   // BATCH UPDATE FACTORS


for (f=0; f<TOTAL_FEATURES; f++) {
   for (i=0; i < num_train_movies; i++) {
       movieId = train_movies_id[i];
     movie_features[movieId - 1][f] += movie_features_gradients[movieId - 1][f];
    }
    for (i=0; i<TOTAL_CUSTOMERS; i++) {
      cust_features[i][f] += cust_features_gradients[i][f];
    }
  }


 for (i=0; i < num_train_movies; i++) {
     movieId = train_movies_id[i];
    m_bias[movieId - 1] += m_bias_gradients[movieId - 1];
  }

  for (i=0; i<TOTAL_CUSTOMERS; i++) {
    c_bias[i] += c_bias_gradients[i];
  }

float p_array[TOTAL_PROBES];

   for (i=0; i < TOTAL_PROBES; i++) {

      movieId = probe_movies[i];
      custId = probe_customers[i];

      // Predict rating and calc error
      p = predict_svd_rating (movieId, custId, TOTAL_FEATURES);

      p = (rating_range*p + min_r);

	p_array[i] = p;

      err = ((double)probe_real_scores[i] - p);

      probe_sq += err*err;

    }


    // stop timer and display time
    stop_e = time(NULL);
    diff = difftime(stop_e, start_e);

    rmse = sqrt(sq/TOTAL_RATES);
    probe_rmse = sqrt(probe_sq/TOTAL_PROBES);
    epochs_probe_error[e]=probe_rmse;

	//////Print out calculation

	if (probe_rmse < MIN_ERROR) {

	MIN_ERROR = probe_rmse;


	// Open file to store probes
		char probes_file[80];
		char str_features[20];

		sprintf(str_features,"%d",TOTAL_FEATURES);

		strcpy (probes_file, str_features);
		strcat (probes_file, "-");
		strcat (probes_file, mysqlD.database);
		strcat (probes_file, "-");
		strcat (probes_file, algorithm_name);
		strcat (probes_file, ".txt");

		FILE *fp = fopen(probes_file,"w");

	   for (i=0; i < TOTAL_PROBES; i++) {

		//Write data to file
		fprintf(fp,"%d,%d,%d,%f\n",probe_customers[i],probe_movies[i],probe_real_scores[i],p_array[i]);
		}

		//close file
		fclose(fp);

	}

	avg_diff += diff;
    
  }
 

  array_min(epochs_probe_error, &final_probe_rmse, &final_epochs_for_probe, MAX_EPOCHS);
 lg("%f\t\t%d\t\t%f sec\t\t", final_probe_rmse,final_epochs_for_probe,avg_diff/cnt);
/* clean up the database link */
mysql_close(conn);


}


double predict_svd_rating (int movieId, int custId, int TOTAL_FEATURES) {

  int f;
  float sum = 0.0;

  for (f=0; f<TOTAL_FEATURES; f++) {
     sum += movie_features[movieId - 1][f] * cust_features[custId - 1][f];
  }
   sum += c_bias[custId - 1] + m_bias[movieId - 1];

   // *** Add residuals
   sum += GLOBAL_SCALED_AVERAGE;

   return sigmoid(sum);

}

double sigmoid (double alpha) {
  return 1.0/(1.0+exp(-alpha));
}



void lgopen(int argc, char**argv) {
	lgfile=fopen("log.txt","a");
	if(!lgfile) error("Cant open log file");
	time_t curtime=time(NULL);
}

void lg(char *fmt,...) {
	char buf[2048];
	va_list ap;

	va_start(ap, fmt);
	vsprintf(buf,fmt,ap);
	va_end(ap);
	fprintf(stderr,"%s",buf);
	if(lgfile) {
		fprintf(lgfile,"%s",buf);
		fflush(lgfile);
	}
}


void error(char *fmt,...) {
	char buf[2048];
	va_list ap;

	va_start(ap, fmt);
	vsprintf(buf,fmt,ap);
	va_end(ap);
	lg("%s",buf);
	lg("\n");
	exit(1);
}

void array_min(double a[], double *min, int *position, int MAX_EPOCHS) {

*min = 9999.0;
*position=-1;

int i;

for (i=0;i<MAX_EPOCHS;i++) {
 if (a[i]<*min){
  *min=a[i];
  *position=i+1;
 }
}
}
