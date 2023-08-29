FROM ann-benchmarks

# https://github.com/pgvector/pgvector/blob/master/Dockerfile

RUN git clone https://github.com/pgvector/pgvector /tmp/pgvector

RUN DEBIAN_FRONTEND=noninteractive apt-get -y install tzdata
RUN apt-get update && apt-get install -y --no-install-recommends build-essential postgresql postgresql-server-dev-all
RUN sh -c 'echo "local all all trust" > /etc/postgresql/14/main/pg_hba.conf'
RUN cd /tmp/pgvector && \
	make clean && \
	make OPTFLAGS="-march=native -mprefer-vector-width=512" && \
	make install

USER postgres
RUN service postgresql start && \
    psql -c "CREATE USER ann WITH ENCRYPTED PASSWORD 'ann'" && \
    psql -c "CREATE DATABASE ann" && \
    psql -c "GRANT ALL PRIVILEGES ON DATABASE ann TO ann" && \
    psql -d ann -c "CREATE EXTENSION vector" && \
    psql -d ann -c "CREATE EXTENSION pg_prewarm" && \
    psql -c "ALTER USER ann SET maintenance_work_mem = '4GB'" && \
    psql -c "ALTER SYSTEM SET shared_buffers = '4GB'" && \
    psql -c "ALTER SYSTEM SET effective_cache_size = '8GB'" && \
    psql -c "ALTER SYSTEM SET max_wal_size = '20GB'" && \
    psql -c "ALTER SYSTEM SET jit = 'off'"
USER root

RUN pip install psycopg[binary] pgvector
