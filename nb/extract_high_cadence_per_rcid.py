#!/usr/bin/env python3

from __future__ import annotations

from subprocess import call, check_call, check_output


TABLE = 'ztf.dr17_olc'
DOCKER_CMD = [
    'docker',
    'exec',
    '-t',
    'clickhouse',
]
RCIDS = list(range(64))


def outfile_name(field: int, rcid: int):
    return f"/output/high_cadence_{field:04d}-{rcid:02d}.parquet"


def run_query(query: str):
    output = check_output(
        DOCKER_CMD + [
            'clickhouse',
            'client',
            '--query',
            query
        ],
    )
    return output.decode().splitlines()


def get_all_fields() -> list[int]:
    output = run_query(f'SELECT fieldid FROM {TABLE} GROUP BY fieldid ORDER BY fieldid')
    fields = list(map(int, output))
    return fields


def valid_file_exists(field: int, rcid: int) -> bool:
    name = outfile_name(field, rcid)
    if call(
            DOCKER_CMD + [
                'test',
                '-s',
                name,
            ]
    ) != 0:
        check_call(
            DOCKER_CMD + [
                'rm',
                '-f',
                name,
            ]
        )
        return False
    return True


def extract(field: int, rcid: int):
    query = f'''
WITH {field} AS target_field, {rcid} AS target_rcid
SELECT oid, start_mjd, groupArray(mjd) AS mjd, groupArray(mag) AS mag, groupArray(magerr) AS magerr, groupArray(expid) AS expid, groupArray(fwhm) AS fwhm, any(reduced_chi_square) AS reduced_chi_square, any(filter) AS filter, any(len) AS original_len
FROM (
SELECT oid, start_mjd, mjd, mag, magerr, reduced_chi_square, expid, fwhm, filter, len
FROM (
SELECT oid,
       split.2 as mjd,
       split.3 as mag,
       split.4 as magerr,
       mjd[1] AS start_mjd,
       reduced_chi_square,
       fieldid,
       rcid,
       filter,
       len
FROM (
      SELECT arraySum(arrayMap((a, b) -> a / power(b, 2), split.3, split.4)) /
             arraySum((a) -> 1 / power(a, 2), split.4)            AS weighted_mean_value,
             arrayMap(a -> a - weighted_mean_value, split.3)      AS diff,
             arrayMap((a, b) -> power((a / b), 2), diff, split.4) AS part,
             len,
             arraySum(part) / (len - 1)                           as reduced_chi_square,
             split,
             oid,
             fieldid,
             rcid,
             filter
      FROM (
            SELECT split[len].2 - split[1].2                               AS duration,
                   arrayJoin(arraySplit(x -> x.1 > 0.5 / 24, light_curve)) AS split,
                   length(split)                                           AS len,
                   oid,
                   fieldid,
                   rcid,
                   filter
            FROM (
                  SELECT oid,
                         arraySort(mjd)                                         AS mjd_time,
                         arraySort((m, t) -> t, mag, mjd)                       AS mag_array,
                         arraySort((m, t) -> t, magerr, mjd)                    AS err_array,
                         arrayDifference(mjd_time)                              AS mjd_arr_diff,
                         arrayZip(mjd_arr_diff, mjd_time, mag_array, err_array) AS light_curve,
		         fieldid,
			 rcid,
                         filter
                  FROM {TABLE}
                  WHERE fieldid = target_field AND rcid = target_rcid
                     )
            WHERE len >= 10
              and duration > 0.5 / 24
               )
         )
WHERE reduced_chi_square > 3
) AS dr
ARRAY JOIN mjd, mag, magerr
JOIN ztf.exposures AS exp ON dr.fieldid = exp.field AND dr.rcid = exp.rcid WHERE abs(dr.mjd - exp.expmid_hmjd) <= 15.0 / 86400.0
)
GROUP BY oid, start_mjd
INTO OUTFILE '{outfile_name(field, rcid)}' FORMAT Parquet
SETTINGS max_bytes_before_external_group_by={32 * (1 << 30)}, max_memory_usage={96 * (1 << 30)}
'''
    run_query(query)


def main():
    print('Getting field list')
    fields = get_all_fields()
    
    for i, field in enumerate(fields):
        print(f'Extracting field {i+1}/{len(fields)}: {field}')
        for rcid in RCIDS:
            print(f'    rcid {rcid}')
            if valid_file_exists(field, rcid):
                print('        Valid file exists, skipping')
                continue
            extract(field, rcid)


if __name__ == '__main__':
    main()
