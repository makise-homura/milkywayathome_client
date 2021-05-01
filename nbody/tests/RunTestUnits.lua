
require "NBodyTesting"
require "persistence"

local arg = {...}

assert(#arg == 5, "Test driver expected 5 arguments got " .. #arg)

local nbodyBinary = arg[1]
local testDir = arg[2]
local testName = arg[3]
local histogramName = arg[4]
local testBodies = arg[5]

local nbodyFlags = getExtraNBodyFlags()
eprintf("NBODY_FLAGS = %s\n", nbodyFlags)

math.randomseed(os.time())

-- Pick one of the random seeds used in generating these tests
local testSeeds = { "670828913", "886885833", "715144259", "430281807", "543966758" }
--local testSeed = testSeeds[math.random(1, #testSeeds)]
local testSeed = testSeeds[3]


refResults = {
   ["model_1"] = {
      ["100"] = {
         ["670828913"] = 2933.684409891683,
         ["886885833"] = 2942.472887150486,
         ["715144259"] = 2922.4523932402035,
         ["430281807"] = 3008.9259355031068,
         ["543966758"] = 2942.4431251631527
      },

      ["1024"] = {
         ["670828913"] = 500.35228759316124,
         ["886885833"] = 1069.4100882167183,
         ["715144259"] = 548.2199872564589,
         ["430281807"] = 335.4631058468563,
         ["543966758"] = 778.585813552342
      },

      ["10000"] = {
         ["670828913"] = 515.1456066074282,
         ["886885833"] = 667.0434231253009,
         ["715144259"] = 553.7379313982775,
         ["430281807"] = 573.6180423281077,
         ["543966758"] = 608.0932197636183
      }
   },

   ["model_2"] = {
      ["100"] = {
         ["670828913"] = 2997.75336907341,
         ["886885833"] = 3001.720060514301,
         ["715144259"] = 2821.066411538379,
         ["430281807"] = 3304.5785512770376,
         ["543966758"] = 3292.8443190880225
      },

      ["1024"] = {
         ["670828913"] = 2291.4638113764913,
         ["886885833"] = 2471.174641475892,
         ["715144259"] = 2341.520264404923,
         ["430281807"] = 2396.6999175281553,
         ["543966758"] = 2791.9349437238056
      },

      ["10000"] = {
         ["670828913"] = 1854.0840557122579,
         ["886885833"] = 2050.6292208329755,
         ["715144259"] = 1992.1604335113364,
         ["430281807"] = 2018.93580972266,
         ["543966758"] = 1974.21634352305
      }
   },

   ["model_3"] = {
      ["100"] = {
         ["670828913"] = 2977.617211132589,
         ["886885833"] = 3022.113142338964,
         ["715144259"] = 2921.1819286256855,
         ["430281807"] = 3237.0129951622557,
         ["543966758"] = 3100.6920404152766
      },

      ["1024"] = {
         ["670828913"] = 2499.932322599812,
         ["886885833"] = 3144.6668198182215,
         ["715144259"] = 2904.6350301877997,
         ["430281807"] = 2975.5097366756772,
         ["543966758"] = 2824.5423174450143
      },

      ["10000"] = {
         ["670828913"] = 2825.0095531483744,
         ["886885833"] = 2983.847786066148,
         ["715144259"] = 2879.2735728603716,
         ["430281807"] = 2915.650150728656,
         ["543966758"] = 2809.0106839966843
      }
   },

   ["model_4"] = {
      ["100"] = {
         ["670828913"] = 2940.75732413997,
         ["886885833"] = 2939.4974030690014,
         ["715144259"] = 2937.2254533587975,
         ["430281807"] = 2784.5381625396226,
         ["543966758"] = 2952.0600000076693
      },

      ["1024"] = {
         ["670828913"] = 593.6265775213317,
         ["886885833"] = 646.9761105880752,
         ["715144259"] = 485.2764724777679,
         ["430281807"] = 514.7333458951916,
         ["543966758"] = 636.7772042601774
      },

      ["10000"] = {
         ["670828913"] = 498.53305485662486,
         ["886885833"] = 598.4918269007578,
         ["715144259"] = 545.43153392973,
         ["430281807"] = 514.340837916246,
         ["543966758"] = 494.75439983036614
      }
   },

   ["model_5"] = {
      ["100"] = {
         ["670828913"] = 2945.3862880888364,
         ["886885833"] = 2939.513919774645,
         ["715144259"] = 2951.1587630111408,
         ["430281807"] = 2902.7052137670157,
         ["543966758"] = 2922.8485543218194
      },

      ["1024"] = {
         ["670828913"] = 467.3839190263135,
         ["886885833"] = 1380.250632450012,
         ["715144259"] = 875.8125984274415,
         ["430281807"] = 1104.0326614973767,
         ["543966758"] = 1106.796442531717
      },

      ["10000"] = {
         ["670828913"] = 434.5832462800824,
         ["886885833"] = 591.3799999822468,
         ["715144259"] = 467.40873541576804,
         ["430281807"] = 462.1888842964611,
         ["543966758"] = 586.7583746895901
      }
   },

   ["model_5_bounds_test"] = {
      ["100"] = {
         ["670828913"] = 5946.28433617035,
         ["886885833"] = 6036.98417139523,
         ["715144259"] = 6060.400968251252,
         ["430281807"] = 5997.736403489975,
         ["543966758"] = 5992.546357198306
      },

      ["1024"] = {
         ["670828913"] = 6019.036587887985,
         ["886885833"] = 6050.293569052645,
         ["715144259"] = 6035.202611387617,
         ["430281807"] = 6025.35130045436,
         ["543966758"] = 6039.557780078374
      },

      ["10000"] = {
         ["670828913"] = 4172.891144508808,
         ["886885833"] = 4190.15105004247,
         ["715144259"] = 4313.0935458276945,
         ["430281807"] = 4345.813687825735,
         ["543966758"] = 4382.760288373556
      }
   },

   ["model_6"] = {
      ["100"] = {
         ["670828913"] = 2939.6317142843905,
         ["886885833"] = 2932.415444894173,
         ["715144259"] = 2937.5672026492953,
         ["430281807"] = 2934.84438293248,
         ["543966758"] = 2928.1269261461457
      },

      ["1024"] = {
         ["670828913"] = 417.57553522528076,
         ["886885833"] = 251.5311454473244,
         ["715144259"] = 464.439952554486,
         ["430281807"] = 395.11793855992806,
         ["543966758"] = 593.1255820499638
      },

      ["10000"] = {
         ["670828913"] = 374.092577033276,
         ["886885833"] = 354.0947004654817,
         ["715144259"] = 363.47540673832475,
         ["430281807"] = 333.06334651629163,
         ["543966758"] = 306.41616888397925
      }
   },

   ["model_7"] = {
      ["100"] = {
         ["670828913"] = 2944.771467686517,
         ["886885833"] = 2931.9400342893473,
         ["715144259"] = 2988.4288646523814,
         ["430281807"] = 2949.417040827505,
         ["543966758"] = 2845.6270244061866
      },

      ["1024"] = {
         ["670828913"] = 373.9652576506462,
         ["886885833"] = 1149.9939103504116,
         ["715144259"] = 1112.5085717905008,
         ["430281807"] = 378.6008221045786,
         ["543966758"] = 1193.0462395339532
      },

      ["10000"] = {
         ["670828913"] = 425.29903224135,
         ["886885833"] = 551.8503442208852,
         ["715144259"] = 525.5863985061176,
         ["430281807"] = 494.38709238531385,
         ["543966758"] = 512.2438180687298
      }
   },

   ["model_8"] = {
      ["100"] = {
         ["670828913"] = 2935.6913058392774,
         ["886885833"] = 2932.533205710029,
         ["715144259"] = 2936.83517710901,
         ["430281807"] = 2953.066894327295,
         ["543966758"] = 2933.845470569983
      },

      ["1024"] = {
         ["670828913"] = 601.5943961418791,
         ["886885833"] = 412.5306338178487,
         ["715144259"] = 493.3761827486983,
         ["430281807"] = 661.3813140611503,
         ["543966758"] = 469.23334329145894
      },

      ["10000"] = {
         ["670828913"] = 287.1501601900975,
         ["886885833"] = 351.3259595733631,
         ["715144259"] = 328.0573235907291,
         ["430281807"] = 317.24489315753374,
         ["543966758"] = 327.946011478938
      }
   },

   ["model_9"] = {
      ["100"] = {
         ["670828913"] = 2947.2283422958894,
         ["886885833"] = 2957.943212612583,
         ["715144259"] = 2928.134126839998,
         ["430281807"] = 2931.1874024030776,
         ["543966758"] = 2942.745744585081
      },

      ["1024"] = {
         ["670828913"] = 944.1396968533672,
         ["886885833"] = 1151.6765519766818,
         ["715144259"] = 591.9249599905268,
         ["430281807"] = 544.160827145035,
         ["543966758"] = 1131.9731447234878
      },

      ["10000"] = {
         ["670828913"] = 121.84604237160457,
         ["886885833"] = 133.47005848198987,
         ["715144259"] = 119.56074934348257,
         ["430281807"] = 118.35372182887193,
         ["543966758"] = 142.75748764813764
      }
   },

   ["model_ninkovic"] = {
      ["100"] = {
         ["670828913"] = 2937.27054562748,
         ["886885833"] = 2939.6205819533507,
         ["715144259"] = 2940.933239598961,
         ["430281807"] = 2941.7857103740316,
         ["543966758"] = 2941.6596516564437
      },

      ["1024"] = {
         ["670828913"] = 767.7708506207632,
         ["886885833"] = 736.1972030499809,
         ["715144259"] = 915.4242581670803,
         ["430281807"] = 553.1415030660112,
         ["543966758"] = 771.6911772015821
      },

      ["10000"] = {
         ["670828913"] = 341.00513973789333,
         ["886885833"] = 454.480097245125,
         ["715144259"] = 397.5169620225546,
         ["430281807"] = 449.2762684957829,
         ["543966758"] = 372.71759870232654
      }
   },

   ["model_triaxial"] = {
      ["100"] = {
         ["670828913"] = 2950.156751273609,
         ["886885833"] = 2987.6935609154652,
         ["715144259"] = 2988.182166337473,
         ["430281807"] = 2951.075109090531,
         ["543966758"] = 3007.0352384974294
      },

      ["1024"] = {
         ["670828913"] = 983.8462367748019,
         ["886885833"] = 1630.30736923363,
         ["715144259"] = 934.3122483281386,
         ["430281807"] = 1157.9920515292438,
         ["543966758"] = 1610.1340195536495
      },

      ["10000"] = {
         ["670828913"] = 967.6800626588107,
         ["886885833"] = 1062.024874394489,
         ["715144259"] = 940.5591629318332,
         ["430281807"] = 1072.461238958144,
         ["543966758"] = 969.3201674011589
      }
   },

   ["model_newhist1"] = {
      ["100"] = {
         ["670828913"] = 8244546.04713184,
         ["886885833"] = 6536922.078011205,
         ["715144259"] = 5359691.175415518,
         ["430281807"] = 7500604.082941535,
         ["543966758"] = 6815924.5900467625
      },

      ["1024"] = {
         ["670828913"] = 3589963.3865490737,
         ["886885833"] = 1862294.9382853184,
         ["715144259"] = 2144332.132103072,
         ["430281807"] = 1412856.9254726083,
         ["543966758"] = 1779683.246461625
      },

      ["10000"] = {
         ["670828913"] = 2594.5990167479554,
         ["886885833"] = 2371.1200399379786,
         ["715144259"] = 2671.156141715882,
         ["430281807"] = 2546.1832504252543,
         ["543966758"] = 2651.818241628261
      }
   },

   ["model_newhist2"] = {
      ["100"] = {
         ["670828913"] = 4886433.248192644,
         ["886885833"] = 6846588.081505293,
         ["715144259"] = 6238544.80718608,
         ["430281807"] = 8224776.656258929,
         ["543966758"] = 5805520.575577043
      },

      ["1024"] = {
         ["670828913"] = 2182093.2478145435,
         ["886885833"] = 1318181.8469775545,
         ["715144259"] = 3422712.8621063842,
         ["430281807"] = 1318385.3052691687,
         ["543966758"] = 1526079.2723780444
      },

      ["10000"] = {
         ["670828913"] = 828.8396237460306,
         ["886885833"] = 1244.3515940725663,
         ["715144259"] = 1286.0848933952955,
         ["430281807"] = 1165.2777366226899,
         ["543966758"] = 1045.542170424484
      }
   },

   ["model_newhist3"] = {
      ["100"] = {
         ["670828913"] = 4691427.973849812,
         ["886885833"] = 5766491.395081397,
         ["715144259"] = 6024038.959919094,
         ["430281807"] = 3658438.5460254033,
         ["543966758"] = 7597442.829448655
      },

      ["1024"] = {
         ["670828913"] = 1882531.8233966578,
         ["886885833"] = 1540431.9994866205,
         ["715144259"] = 3017742.910013158,
         ["430281807"] = 1859783.9753114097,
         ["543966758"] = 1592977.90529353
      },

      ["10000"] = {
         ["670828913"] = 2638.733070722022,
         ["886885833"] = 2478.1171174698825,
         ["715144259"] = 2400.998653183547,
         ["430281807"] = 2580.228843157575,
         ["543966758"] = 2328.119910527963
      }
   },

   ["model_LMC"] = {
      ["100"] = {
         ["670828913"] = 3156.2824411476977,
         ["886885833"] = 3078.846616595685,
         ["715144259"] = 3120.916352044287,
         ["430281807"] = 3207.318399752033,
         ["543966758"] = 3133.61065546585
      },

      ["1024"] = {
         ["670828913"] = 2584.601153951753,
         ["886885833"] = 2906.761414507592,
         ["715144259"] = 2846.3367755936133,
         ["430281807"] = 2657.8866985013487,
         ["543966758"] = 2799.621432776491
      },

      ["10000"] = {
         ["670828913"] = 3664.0704009454794,
         ["886885833"] = 3851.1737077874386,
         ["715144259"] = 3685.0913282509027,
         ["430281807"] = 3669.3836949918227,
         ["543966758"] = 3801.8121710489186
      }
   },

   ["model_bar"] = {
      ["100"] = {
         ["670828913"] = 2866.770617953743567,
         ["886885833"] = 2964.293392996516104,
         ["715144259"] = 2811.497972843205844,
         ["430281807"] = 3028.133681549090852,
         ["543966758"] = 2940.887816285809549
      },

      ["1024"] = {
         ["670828913"] = 2166.595784028842445,
         ["886885833"] = 2026.073019969936013,
         ["715144259"] = 2237.598630754825535,
         ["430281807"] = 2418.846607513595245,
         ["543966758"] = 2192.723938809735046
      },

      ["10000"] = {
         ["670828913"] = 555.642878927146057,
         ["886885833"] = 512.486718191362115,
         ["715144259"] = 567.035043289565692,
         ["430281807"] = 527.399858237311491,
         ["543966758"] = 688.834062409447142
      }
   },

   ["model_LMC_bar"] = {
      ["100"] = {
         ["670828913"] = 3155.123615576813791,
         ["886885833"] = 3130.039253899083633,
         ["715144259"] = 3020.141759582297254,
         ["430281807"] = 3211.791964840703713,
         ["543966758"] = 3066.803331267471549
      },

      ["1024"] = {
         ["670828913"] = 2572.225069142848952,
         ["886885833"] = 2733.134436584396099,
         ["715144259"] = 2630.993118606981625,
         ["430281807"] = 2597.491708837917486,
         ["543966758"] = 2994.530783209523634
      },

      ["10000"] = {
         ["670828913"] = 3802.078768977051368,
         ["886885833"] = 3835.161425252691515,
         ["715144259"] = 3678.915394318895778,
         ["430281807"] = 3644.857581912348905,
         ["543966758"] = 3822.271253710230667
      }
   }
}



function resultCloseEnough(a, b)
   return math.abs(a - b) < 1.0e-10
end

errFmtStr = [[
Result differs from expected:
   Expected = %20.15f  Actual = %20.15f  |Difference| = %20.15f
]]

function runCheckTest(testName, histogram, seed, nbody, ...)
   local fileResults, bodyResults
   local ret, result

   if not generatingResults then
      -- Check if the result exists first so we don't waste time on a useless test
      fileResults = assert(refResults[testName], "Didn't find result for test file")
      bodyResults = assert(fileResults[nbody], "Didn't find result with matching bodies")
      refResult = assert(bodyResults[seed], "Didn't find result with matching seed")
   end

   --eprintf("CHECKTEST - Before runFullTest\n")

   ret = runFullTest{
      nbodyBin  = nbodyBinary,
      testDir   = testDir,
      testName  = testName,
      histogram = histogram,
      seed      = seed,
      cached    = false,
      extraArgs = { nbody }
   }

   --eprintf(ret.."\n")
   --eprintf("CHECKTEST - Before findLikelihood\n")

   result = findLikelihood(ret, false)

   --eprintf("CHECKTEST - Before write(ret)\n")

   io.stdout:write(ret)

   if generatingResults then
      io.stderr:write(string.format("Test result: %d, %d, %s: %20.15f\n", nbody, seed, testName, result))
      return false
   end

   if result == nil then
      return true
   end

   --eprintf("CHECKTEST - Before notClose\n")

   local notClose = not resultCloseEnough(refResult, result)
   if notClose then
      io.stderr:write(string.format(errFmtStr, refResult, result, math.abs(result - refResult)))
   end

   return notClose
end

-- return true if passed
function testProbabilistic(resultFile, testName, histogram, nbody, iterations)
   local testTable, histTable, answer
   local resultTable = persisence.load(resultFile)
   assert(resultTable, "Failed to open result file " .. resultFile)

   testTable = assert(resultTable[testName], "Did not find result for test " .. testName)
   histTable = assert(testTable[nbody], "Did not find result for nbody " .. tostring(nbody))
   answer = assert(histTable[nbody], "Did not find result for histogram " .. histogram)

   local minAccepted = answer.mean - 3.0 * answer.stddev
   local maxAccepted = answer.mean + 3.0 * answer.stddev

   local result = 0.0
   local z = (result - answer.mean) / answer.stddev


   return true
end



function getResultName(testName)
   return string.format("%s__results.lua", testName)
end

if runCheckTest(testName, histogramName, testSeed, testBodies) then
   os.exit(1)
end


