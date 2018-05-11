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
#define MAX_EPOCHS       10                  // Max epochs per feature
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


#define W_INIT_SEED      0.1
#define W_INIT_VARIANCE  0.01
#define W_INIT           (W_INIT_SEED + (2.0*(rand()/(float)(RAND_MAX)) - 1.0)*W_INIT_VARIANCE) //


double LRATE1u  =            0.003;        // Learning rate parameter for features
double LAMDA1u  =            0.015;        // reg for features
double LRATE1m  =            0.003;        // Learning rate parameter for features
double LAMDA1m  =            0.5;        // reg for features

double  LRATE3  =            0.003;        // Learning rate parameter for weights

double LRATE2ub =            0.003;        // Learning rate parameter for biases
double LAMDA2ub =            0.015;        // reg for biases
double LRATE2mb =            0.003;        // Learning rate parameter for biases
double LAMDA2mb =            0.015;          // reg for biases

double LAMDA3 =              1.0;               // Regularization
 


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


float randn(void);

void float_array_fill_zeros (float *my_array, unsigned int size_of_my_array);

float predict_svd_rating (int movieId, int custId, int TOTAL_FEATURES);

int rnd(int max);


void create_txt_file ();
void write_txt_file (unsigned int customer_id, unsigned short movie_id);
void close_txt_file ();


void calc_features(int TOTAL_FEATURES);

void calc_user_moviebag (int custId, int *pmovie_id, int num_items, int *c_probe_movies, int c_probe_items, int TOTAL_FEATURES);


void update_user_moviebag (int custId, int *pmovie_id, int num_items, int *c_probe_movies, int c_probe_items,double *newpu, double *oldpu, int TOTAL_FEATURES);


void calc_users_moviebag(int *ptrain_users, int num_users, int TOTAL_FEATURES);


FILE *lgfile=NULL;
void lg(char *fmt,...);
void lgopen(int argc, char**argv);
void error(char *fmt,...);

float final_probe_rmse=0.0;
unsigned int final_epochs_for_probe;


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
float GLOBAL_AVERAGE;

// ****** SVD *********** //
float **movie_features;     // Array of features by movie (using floats to save space)
float **w;  
float **cust_features;   // Array of features by customer (using floats to save space)
float **sum_w;
float *m_bias;
float *c_bias;
float *ei;
// ******************** //

int **user_movies;
int *user_movies_size;
int **user_ratings;

int **user_probe_movies;
int *user_probe_size;

// *** PROBE ***//
int *probe_customers;
int *probe_movies;
int *probe_real_scores;
// *** //



int max_r,min_r;
char algorithm_name[20];

main (int argc, char**argv) {

  lgopen(argc,argv);


  float prediction;
  unsigned int i,h;
  time_t start, stop;
  double diff;
  int TOTAL_FEATURES = atoi(argv[5]);
   /* start timer */
  start = time(NULL);  

  mysqlD.server = argv[1];  // where the mysql database is
  mysqlD.user = argv[2];   // the root user of mysql 
  mysqlD.password = argv[3]; // the password of the root user in mysql
  mysqlD.database = argv[4]; // the databse to pick
 
  // connect to the mysql database
  conn = mysql_connection_setup(mysqlD);


sprintf(query_string,"SELECT count(DISTINCT item_id) FROM ratings");

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


// ****** SVD *********** //


movie_features = ( float** )malloc(TOTAL_MOVIES * sizeof(float *));

  if(movie_features == NULL)
    {
    fprintf(stderr, "out of memory\n");
    exit(-1);
        }
  for(i = 0; i < TOTAL_MOVIES; i++)
    {
    movie_features[i] = ( float* )malloc(TOTAL_FEATURES * sizeof(float));
    if(movie_features[i] == NULL)
      {
      fprintf(stderr, "out of memory\n");
      exit(-1);
      }
    }

w = ( float** )malloc(TOTAL_MOVIES * sizeof(float *));

  if(w == NULL)
    {
    fprintf(stderr, "out of memory\n");
    exit(-1);
        }
  for(i = 0; i < TOTAL_MOVIES; i++)
    {
    w[i] = ( float* )malloc(TOTAL_FEATURES * sizeof(float));
    if(w[i] == NULL)
      {
      fprintf(stderr, "out of memory\n");
      exit(-1);
      }
    }


cust_features = ( float** )malloc(TOTAL_CUSTOMERS * sizeof(float *));

  if(cust_features == NULL)
    {
    fprintf(stderr, "out of memory\n");
    exit(-1);
        }
  for(i = 0; i < TOTAL_CUSTOMERS; i++)
    {
    cust_features[i] = ( float* )malloc(TOTAL_FEATURES * sizeof(float));
    if(cust_features[i] == NULL)
      {
      fprintf(stderr, "out of memory\n");
      exit(-1);
      }
    }


sum_w = ( float** )malloc(TOTAL_CUSTOMERS * sizeof(float *));

  if(sum_w == NULL)
    {
    fprintf(stderr, "out of memory\n");
    exit(-1);
        }
  for(i = 0; i < TOTAL_CUSTOMERS; i++)
    {
    sum_w[i] = ( float* )malloc(TOTAL_FEATURES * sizeof(float));
    if(sum_w[i] == NULL)
      {
      fprintf(stderr, "out of memory\n");
      exit(-1);
      }
    }


m_bias =  (float *)malloc(sizeof(float)*TOTAL_MOVIES);

c_bias =  (float *)malloc(sizeof(float)*TOTAL_CUSTOMERS);

ei =  (float *)malloc(sizeof(float)*TOTAL_CUSTOMERS);


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


user_probe_movies = ( int** )malloc(TOTAL_CUSTOMERS * sizeof(int *));


  if(user_probe_movies == NULL)
    {
    fprintf(stderr, "out of memory for user connections\n");
    exit(-1);
    }


user_probe_size =  (int *)malloc(sizeof(int)*TOTAL_CUSTOMERS);

// ***************** //


/* stop timer and display time */
  stop = time(NULL);
  diff = difftime(stop, start);
//  printf("Defined global arrays: Time elapsed is %f sec\n", diff);


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


/* clean up the database result set */
mysql_free_result(res);

// ******************** //
  

  /* stop timer and display time */
  stop = time(NULL);
  diff = difftime(stop, start);
//  printf("Created Probe arrays: Time elapsed is %f sec\n", diff);

  // start timer
  start = time(NULL);  

  // RUN SVD
//  lg("\n\nCalculating features...\n");

  sscanf(argv[0], "./%s", algorithm_name);
  lg("%s\t\t",algorithm_name);
  calc_features(TOTAL_FEATURES);

  /* stop timer and display time */
  stop = time(NULL);
  diff = difftime(stop, start);
//  printf("\nTrained SVD in %f sec\n", diff);
  lg("%f sec\n", diff);
exit(-1);

  // *** SAVE FEATURES ***
  // lg("\n\nSaving features files...\n");
  //  save_new_features_files();


 // save_predictions();

//////save_residuals();

  // stop timer and display time 
  stop = time(NULL);
  diff = difftime(stop, start);
//  lg("\nPredictions: Time elaspsed is %f sec\n", diff);

  exit(0);
}


//****** SVD *********


void calc_features(int TOTAL_FEATURES) {

  double pu[TOTAL_FEATURES];
  double pudot[TOTAL_FEATURES];
  double pudotdot[TOTAL_FEATURES];

  time_t start, stop, start_e, stop_e;
  double avg_diff=0.0;
  double diff;
int c, d, h,f, e, i,j, custId, cnt = 0;
  
  int num_movies;

  int num_cust_probe_movies;

  double err, p, sq, rmse_last, rmse = 2.0, probe_rmse=9998, probe_rmse_last=9999, probe_sq;
       
  int movieId;
  double cf, mf, wf, cf_bias, mf_bias;

 /* 
  unsigned int startIdx, endIdx;
  unsigned int probeStartIdx, probeEndIdx;  
*/

  // INIT all feature values 
  for (f=0; f<TOTAL_FEATURES; f++) {
    for (i=0; i<TOTAL_MOVIES; i++) {
      movie_features[i][f] = INIT_M;
      w[i][f] = W_INIT;
     // printf("%f\n",movie_features[i][f]);
    }
    for (i=0; i<TOTAL_CUSTOMERS; i++) {
      cust_features[i][f] = INIT_C;
    //  printf("%f\n",cust_features[i][f]);
    }
  }


  // *** INIT biases
  for (i=0; i<TOTAL_MOVIES; i++) {
    m_bias[i] = INIT_Mb;
   // printf("%f\n",m_bias[i]);
  }
  for (i=0; i<TOTAL_CUSTOMERS; i++) {
    c_bias[i] = INIT_Cb;
    //printf("%f\n",c_bias[i]);
  }


////////////////First we count how many users exist in our dataset and store them

sprintf(query_string,"SELECT COUNT(DISTINCT user_id) FROM user_mapping");

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
sprintf(query_string,"SELECT DISTINCT(user_id) FROM user_mapping");

res = mysql_perform_query(conn,query_string);

///fetch all selected rows

 h=0;////just a counter
  while ((row = mysql_fetch_row(res)) !=NULL) {
      train_users_id[h]=atoi(row[0]);
    //printf("%d %d\n",h+1, train_users_id[h]);
      h++;
}
//exit(-1);
 /* clean up the database result set */

mysql_free_result(res);

////////Now we have the train set users stored

//printf("Loading database into memory...\n");

//start = time(NULL);

      for (c=0; c < num_train_users; c++)  {

      custId = train_users_id[c];
      //printf("CustId: %d\n",custId);  


//////Find out how many movies the user have rated

sprintf(query_string,"select count(item_id) FROM train WHERE user_id=%d",custId);

res = mysql_perform_query(conn,query_string);

while ((row = mysql_fetch_row(res)) !=NULL) {
         num_movies=atoi(row[0]);
}

 /* clean up the database result set */
mysql_free_result(res);


//printf("Cust %d #movies %d\n",custId, num_movies);


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


sprintf(query_string,"select item_id, rating_value FROM train WHERE user_id=%d",custId);

//printf("%s\n",query_string);

res = mysql_perform_query(conn,query_string);

 h=0;////just a counter
  while ((row = mysql_fetch_row(res)) !=NULL) {
      user_movies[c][h]=atoi(row[0]);
      user_ratings[c][h]=atoi(row[1]);
      //printf("%d %d\n",user_movies[c][h],user_ratings[c][h]);
      h++;
}

/* clean up the database result set */
mysql_free_result(res);


}


///////We got all movies and ratings for all users


sprintf(query_string,"select count(item_id) FROM probe WHERE user_id=%d",custId);

res = mysql_perform_query(conn,query_string);

while ((row = mysql_fetch_row(res)) !=NULL) {
         num_cust_probe_movies=atoi(row[0]);
}

 /* clean up the database result set */
mysql_free_result(res);

/*
if (num_cust_probe_movies==0) {
printf("%d\n",custId, num_cust_probe_movies);
exit(-1);
}
*/

//printf("%d %d\n",custId, num_cust_probe_movies);

  if (num_cust_probe_movies!=0) {

user_probe_size[c] = num_cust_probe_movies;

user_probe_movies[c] = ( int* )malloc(num_cust_probe_movies * sizeof(int));

    if(user_probe_movies[c] == NULL)
      {
      fprintf(stderr, "out of memory for connections of customer %d\n", custId);
      exit(-1);
      }


//ei[custId-1] = 1.0 / sqrtf( num_movies + num_cust_probe_movies + 1.0);
//ei[custId-1] = 1.0 / sqrtf( num_movies + num_cust_probe_movies);
//printf("%f\n", ei[custId-1]);


sprintf(query_string,"SELECT item_id FROM probe WHERE user_id=%d\n",custId);

res = mysql_perform_query(conn,query_string);

h=0;
while ((row = mysql_fetch_row(res)) !=NULL) {
         user_probe_movies[c][h]=atoi(row[0]);
         h++;
}

//clean up the database result set
mysql_free_result(res);

///////We got all PROBE movies for all users

  }

ei[custId-1] = 1.0 / sqrtf( num_movies + num_cust_probe_movies);

}


//stop = time(NULL);
//diff = difftime(stop,start);
//printf("Data loaded in %f secs\n",diff);


//exit(-1);


// Keep looping until you have stopped making significant (probe_rmse) progress
    while ((probe_rmse < probe_rmse_last - MIN_IMPROVEMENT)) {
    //for (e=0; e < 2; e++) {
        
    start = time(NULL);
    start_e = time(NULL);
    
    cnt++;
    sq = 0;
    probe_sq = 0;
    rmse_last = rmse;
    probe_rmse_last = probe_rmse;

  /////////////////Here starts primary iteration to users

///////continue with the train iteration


     for (c=0; c < num_train_users; c++)  {

      d=c;

      custId = train_users_id[c];
      //printf("%d\n",custId);

      // *** Calc all weights sum for this user

      calc_user_moviebag (custId,user_movies[c],user_movies_size[c], user_probe_movies[c], user_probe_size[c], TOTAL_FEATURES);


      // pu

      for (f=0; f<TOTAL_FEATURES; f++) {      

      pu[f] = ei[custId - 1] * sum_w[custId - 1][f];
      pudot[f]=pu[f];

      }

      if (user_movies_size[c]!=0) {

      for (i=0; i< user_movies_size[c]; i++) {
      //for (i=startIdx; i<=endIdx; i++) 

        movieId=user_movies[c][i];

        //printf("%d %d\n",custId, movieId);
	      //exit(-1);
        
        /*  
        movieId = nf_movies_c_index[i];
        rating = (double) nf_scores_c_index[i];
        */

        // Predict rating and calc error
       // printf("Will call predict\n");
      
        p = predict_svd_rating (movieId, custId, TOTAL_FEATURES);

       // printf("After call to predict\n");

         err = ((float)user_ratings[c][i] - p);

        sq += err*err;


        //printf("custid:%d, movieid:%d, prediction:%f, error:%f\n", custId, movieId, p, err);


        //*** train biases
        cf_bias = c_bias[custId - 1];
        mf_bias = m_bias[movieId - 1];

        //c_bias[custId - 1] += (LRATE2ub * (err - LAMDA1u*LAMDA2ub * cf_bias));
        //m_bias[movieId - 1] += (LRATE2mb * (err - LAMDA1m*LAMDA2mb * mf_bias));          

        c_bias[custId - 1] += (LRATE2ub * (err - LAMDA2ub * cf_bias));
        m_bias[movieId - 1] += (LRATE2mb * (err - LAMDA2mb * mf_bias));
           

        for (f=0; f<TOTAL_FEATURES; f++) {      
                
          // Cache off old feature values
          cf = cust_features[custId - 1][f];
          mf = movie_features[movieId - 1][f];
          pudotdot[f]= pu[f];


         // *** User vector

         cust_features[custId - 1][f] += (LRATE1u * (err * mf - LAMDA1u * cf));
         //printf("C %f ", cust_features[movieId - 1][f]);


           // *** pu

         pu[f] += (LRATE3 * (err *  mf - LAMDA3 * pudotdot[f]));



          // *** Movie vector

         movie_features[movieId - 1][f] += (LRATE1m * (err * (cf + pudotdot[f]) - LAMDA1m * mf));
         //printf("M %.3f ", movie_features[movieId - 1][f]);



/*
        for (j=0; j< num_cust_probe_movies; j++) {

        pu[f] += (LRATE3 * (err * mf - LAMDA3 * pu[f]));

        } 
*/

           //other movies

          // *** Movie weights
          //w[movieId -1][f] += (LRATE3 * (err * ei[custId - 1] * mf - LAMDA3 * wf)); 



          // *** Update Sum of the weights for this user
          //sum_w[custId - 1][f] = sum_w[custId - 1][f] + w[movieId - 1][f] - wf;

        }

          // *** Calc all movie weight sums for all users

      // if (d>30000)
      // printf("%d %d %d\n",d+1,custId,movieId);

     //  printf("\n");

    }

   //  exit(-1);

}

/*
   for (f=0; f<TOTAL_FEATURES; f++) {
      pu[f] = ei[custId - 1] * pu[f];
      }
*/

  update_user_moviebag (custId, user_movies[c], user_movies_size[c], user_probe_movies[c], user_probe_size[c], pu, pudot, TOTAL_FEATURES);


  //  calc_users_moviebag (train_users_id,num_train_users);

      
    if ((d!=0) && (d%10000000 == 0)){

    stop = time(NULL);
    diff = difftime(stop,start);
 //   printf("Done %d Customers in %f secs\n",d,diff);
    start = time(NULL);

   }
  
               
 }


//   printf("Calculating Probe...\n");
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

      movieId = probe_movies[i];
      custId = probe_customers[i];

      // Predict rating and calc error
      p = predict_svd_rating (movieId, custId, TOTAL_FEATURES);

	//Write data to file
	fprintf(fp,"%d,%d,%d,%f\n",custId,movieId,probe_real_scores[i],p);

      err = ((float)probe_real_scores[i] - p);
      probe_sq += err*err;

    }

	//close file
	fclose(fp);

    // stop timer and display time
    stop_e = time(NULL);
    diff = difftime(stop_e, start_e);

    rmse = sqrt(sq/TOTAL_RATES);
    probe_rmse = sqrt(probe_sq/TOTAL_PROBES);

 //   lg("     <set x='%d' y='%f' probe='%f' /> time: %f sec\n", cnt, rmse, probe_rmse, (double) diff);
    
  avg_diff+=diff;

  }

//  printf("\nAverage time spent in each  iteration is %f\n",  avg_diff/cnt);
 
  final_probe_rmse = probe_rmse;
  final_epochs_for_probe = cnt;
 lg("%f\t\t%d\t\t%f sec\t\t", cnt,probe_rmse,avg_diff/cnt);
/* clean up the database link */
mysql_close(conn);


}





float predict_svd_rating (int movieId,  int custId, int TOTAL_FEATURES) {

  int f;
  double sum = 0.0;
    

   for (f=0; f<TOTAL_FEATURES; f++) {

    sum += movie_features[movieId - 1][f] * (cust_features[custId - 1][f] + ei[custId - 1]*sum_w[custId - 1][f]);

  }



  // *** Add biases

  sum += c_bias[custId - 1] + m_bias[movieId - 1];

 
  // *** Add residuals
  sum += GLOBAL_AVERAGE;
  
  if (sum > max_r) sum = (float) max_r;
  if (sum < min_r) sum = (float) min_r;

  return sum;
}


void calc_user_moviebag (int custId, int *pmovie_id, int num_items, int *c_probe_movies, int c_probe_items, int TOTAL_FEATURES) {


  int i, f;

  int movie_id;

  
  for (f=0; f<TOTAL_FEATURES; f++) {



    sum_w[custId - 1][f] = 0.0;

    

    // *** For loop for all the movies she has seen


     for (i=0; i< num_items; i++) {
      //for (i=startIdx; i<=endIdx; i++) 

      movie_id=pmovie_id[i];

      sum_w[custId - 1][f] += w[movie_id - 1][f];

    }


      for (i=0; i< c_probe_items; i++) {
      //for (i=startIdx; i<=endIdx; i++) 

      movie_id=c_probe_movies[i];

      sum_w[custId - 1][f] += w[movie_id - 1][f];

    }

    

  }

}


void update_user_moviebag (int custId, int *pmovie_id, int num_items, int *c_probe_movies, int c_probe_items,double *newpu, double *oldpu, int TOTAL_FEATURES) 
{

 int i, f;

  int movie_id;


  for (f=0; f<TOTAL_FEATURES; f++) {


     sum_w[custId - 1][f] = 0.0;


     for (i=0; i< num_items; i++) {
      //for (i=startIdx; i<=endIdx; i++) 

      movie_id=pmovie_id[i];

      w[movie_id - 1][f] += ei[custId - 1]*(newpu[f]-oldpu[f]);

      sum_w[custId - 1][f] += w[movie_id - 1][f];

    }


      for (i=0; i< c_probe_items; i++) {
      //for (i=startIdx; i<=endIdx; i++) 

      movie_id=c_probe_movies[i];

      w[movie_id - 1][f] += ei[custId - 1]*(newpu[f]-oldpu[f]);

      sum_w[custId - 1][f] += w[movie_id - 1][f];


    }



  }


}


void calc_users_moviebag(int *ptrain_users, int num_users, int TOTAL_FEATURES) {


  int i, c, j, f;

  unsigned int movie_id, cust_id;  

  
//printf("Train users %d\n",num_users);

/*
for (c=0; c < num_users; c++)  {

      printf("%d \n", ptrain_users[c]);
}
*/
//exit(-1);

    for (c=0; c < num_users; c++)  {


    cust_id = ptrain_users[c];


    for (f=0; f<TOTAL_FEATURES; f++) {
      sum_w[cust_id - 1][f] = 0.0;
    }

      // *** For loop for all the movies she has seen
  
sprintf(query_string,"select item_id FROM train WHERE user_id=%d",cust_id);

    res = mysql_perform_query(conn,query_string);

     while ((row = mysql_fetch_row(res)) !=NULL) {

      movie_id=atoi(row[0]);

     for (f=0; f<TOTAL_FEATURES; f++) {
      sum_w[cust_id - 1][f] += w[movie_id - 1][f];
    }

    }

 /* clean up the database result set */
mysql_free_result(res);

    }  

}


void lgopen(int argc, char**argv) {
	lgfile=fopen("log.txt","a");
	if(!lgfile) error("Cant open log file");
//	lg("----------------------------------------------\n");
	/* Print out the date and time in the standard format.  */
	time_t curtime=time(NULL);
//	lg("%s",ctime(&curtime));

//	int i;
//	for(i=0;i<argc;i++)
//		lg("%s ",argv[i]);
//	lg("\n");
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


