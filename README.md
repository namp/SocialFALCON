SocialFALCON is an efficient constrained matrix factorization algorithm for providing recommendations in social rating networks. The algorithm is described in the following paper:

N.Ampazis, T. Emmanouilidis, and F. Sakketou:"A Matrix Factorization Algorithm for Efficient Recommendations in Social Rating Networks Using Constrained Optimization", Machine Learning, 2018. Submitted.

The codes provided here implement all the algorithms that are benchmarked in the experimental section of the paper on the Epinions and Flixter datasets.


-------Installation Instructions-------

A C compiler is needed to compile code for your system. Apart from standard C libraries, mysql.h must be installed and properly linked to compile code.

For Ubuntu systems, libmysqlclient-dev package must be installed.

Install using the following command:

sudo apt-get install libmysqlclient-dev

Then you can compile code using gcc issuing the command:

gcc [algorithm_name].c -o regsvd -I/usr/include/mysql -L/usr/lib/mysql -lmysqlclient

-------------Running-------------------

Each algorithm takes as command line arguments the following variables:

MySQL server name or ip, database user, database user password, database name, number of features

The SocialMF algorithm also needs one more argument which is the max number of epochs it will run.

Each algorithm outputs a results csv file named [number of features]-[database name]-[algorithm name].txt with fields:

user_id,item_id,real_rating,predicted_rating

and appends overall results to a log.txt file created in the same directory of algorithms executable.

Results csv file can be processed to extract cold user results or other statistics.

--------Database Installation---------

Datasets are available for download from here (http://labs.fme.aegean.gr/ideal/socialfalcon-datasets/). For every dataset (Epinions, Flixter) we provide 5 different cross-validation (CV)splits of train/evaluation sets. 

You can create the tables using the MySQL commands we provide and insert data using corresponding csv file. We used one database for each fold.

For example, you can use the following procedure to create and load the Epinions ratings table:

First you have to create a database to store tables, for example "Epinions"

Then you have to isue this command to enter a MySQL terminal:

mysql --local-infile=1 -u root(or some user) -p
use Epinions

Then this sql command to create the table:

CREATE TABLE IF NOT EXISTS `ratings` (
`user_id` int(11) NOT NULL,
`item_id` int(11) DEFAULT NULL,
`rating_value` int(11) NOT NULL,
KEY `item_id` (`item_id`),
KEY `user_id` (`user_id`)
) ENGINE=MyISAM DEFAULT CHARSET=utf8;

and finally this to insert data:

load data local infile 'Epinions_ratings.csv' INTO table ratings fields terminated by ';' ENCLOSED BY '"';

You should do also do this for the train and validation (probe) sets of each of the 5-fold CV spits of both datasets.


