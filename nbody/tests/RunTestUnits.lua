
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
local testSeed = testSeeds[4]


refResults = {
   ["model_1"] = {
      ["100"] = {
         ["670828913"] = 2933.684409891683117,
         ["886885833"] = 2942.149152135481472,
         ["715144259"] = 2922.452393240203492,
         ["430281807"] = 2984.351725528127645,
         ["543966758"] = 2957.529678073099603
      },

      ["1024"] = {
         ["670828913"] = 531.180057242967109,
         ["886885833"] = 925.703641342560331,
         ["715144259"] = 541.917450204852457,
         ["430281807"] = 589.150613678012178,
         ["543966758"] = 555.904043186897752
      },

      ["10000"] = {
         ["670828913"] = 515.991821244582297,
         ["886885833"] = 657.075638670085141,
         ["715144259"] = 555.195340290395507,
         ["430281807"] = 486.437913717118420,
         ["543966758"] = 499.318391776659894
      }
   },

   ["model_2"] = {
      ["100"] = {
         ["670828913"] = 2997.293543273928208,
         ["886885833"] = 3001.720060514301167,
         ["715144259"] = 2821.301637929617755,
         ["430281807"] = 3033.485643232966140,
         ["543966758"] = 2956.509253941507723
      },

      ["1024"] = {
         ["670828913"] = 2293.634214970382345,
         ["886885833"] = 2426.452409747164893,
         ["715144259"] = 2350.526319177436562,
         ["430281807"] = 1946.281936394002059,
         ["543966758"] = 1838.058280123110308
      },

      ["10000"] = {
         ["670828913"] = 1915.783374662063352,
         ["886885833"] = 2013.727322467846534,
         ["715144259"] = 1961.445571681523461,
         ["430281807"] = 1700.511927508822964,
         ["543966758"] = 1782.113128240187507
      }
   },

   ["model_3"] = {
      ["100"] = {
         ["670828913"] = 2975.883942901933551,
         ["886885833"] = 3017.221621357173262,
         ["715144259"] = 3014.676965030997053,
         ["430281807"] = 3104.027332807339462,
         ["543966758"] = 3047.261100671892109
      },

      ["1024"] = {
         ["670828913"] = 2641.571119808188996,
         ["886885833"] = 2981.219605745454373,
         ["715144259"] = 2747.254196130287710,
         ["430281807"] = 2155.533497151694519,
         ["543966758"] = 2005.147817619336593
      },

      ["10000"] = {
         ["670828913"] = 2799.385786815821575,
         ["886885833"] = 3006.463006736485568,
         ["715144259"] = 2827.801053578225947,
         ["430281807"] = 2705.266327871026533,
         ["543966758"] = 2634.850297646053605
      }
   },

   ["model_4"] = {
      ["100"] = {
         ["670828913"] = 2940.757324139970024,
         ["886885833"] = 2935.676669637463419,
         ["715144259"] = 2937.225453358797495,
         ["430281807"] = 2941.587147308531712,
         ["543966758"] = 2944.754987680415525
      },

      ["1024"] = {
         ["670828913"] = 435.818492429694118,
         ["886885833"] = 641.652792371019700,
         ["715144259"] = 553.871291333205818,
         ["430281807"] = 557.158238400716982,
         ["543966758"] = 597.210953153113564
      },

      ["10000"] = {
         ["670828913"] = 497.669239320331997,
         ["886885833"] = 604.670230044777895,
         ["715144259"] = 548.668784861488120,
         ["430281807"] = 506.753579767947315,
         ["543966758"] = 435.823440755992181
      }
   },

   ["model_5"] = {
      ["100"] = {
         ["670828913"] = 2945.386288088836409,
         ["886885833"] = 2939.152313173569382,
         ["715144259"] = 2951.158763011140763,
         ["430281807"] = 2943.185667639507756,
         ["543966758"] = 2951.949694115800412
      },

      ["1024"] = {
         ["670828913"] = 512.434639762705842,
         ["886885833"] = 1297.180293207168916,
         ["715144259"] = 874.050354416460777,
         ["430281807"] = 848.620116691475573,
         ["543966758"] = 761.820242386376549
      },

      ["10000"] = {
         ["670828913"] = 431.217127556717287,
         ["886885833"] = 593.096391156776804,
         ["715144259"] = 470.169472062001773,
         ["430281807"] = 510.257464082453566,
         ["543966758"] = 583.239648753696770
      }
   },

   ["model_5_bounds_test"] = {
      ["100"] = {
         ["670828913"] = 5940.352232491616633,
         ["886885833"] = 6048.892412754213183,
         ["715144259"] = 6079.363584714976241,
         ["430281807"] = 5978.106905539169929,
         ["543966758"] = 5920.388817218797158
      },

      ["1024"] = {
         ["670828913"] = 6067.291433578061515,
         ["886885833"] = 5961.931071283683195,
         ["715144259"] = 6036.172870815794340,
         ["430281807"] = 6044.466410610495586,
         ["543966758"] = 6056.142462394738686
      },

      ["10000"] = {
         ["670828913"] = 4176.743993879557820,
         ["886885833"] = 4190.732995355771891,
         ["715144259"] = 4311.132190361318862,
         ["430281807"] = 4361.944085202775568,
         ["543966758"] = 4253.087871714278663
      }
   },

   ["model_6"] = {
      ["100"] = {
         ["670828913"] = 2939.979456598023262,
         ["886885833"] = 2932.415444894173106,
         ["715144259"] = 2937.567202649295268,
         ["430281807"] = 2928.860281832800865,
         ["543966758"] = 2945.258485804310112
      },

      ["1024"] = {
         ["670828913"] = 526.933748861328809,
         ["886885833"] = 275.266226049569468,
         ["715144259"] = 406.325638601272374,
         ["430281807"] = 443.310809221892782,
         ["543966758"] = 465.178854713008946
      },

      ["10000"] = {
         ["670828913"] = 378.353965502478559,
         ["886885833"] = 354.206597696954418,
         ["715144259"] = 365.371179570433924,
         ["430281807"] = 345.917698345749045,
         ["543966758"] = 318.205077710931732
      }

   },

   ["model_7"] = {
      ["100"] = {
         ["670828913"] = 2944.771467686517099,
         ["886885833"] = 2931.940034289347295,
         ["715144259"] = 2988.428864652381435,
         ["430281807"] = 2928.372183543255687,
         ["543966758"] = 2921.421816868545648
      },

      ["1024"] = {
         ["670828913"] = 321.269237011866949,
         ["886885833"] = 1019.626901392249579,
         ["715144259"] = 1264.233947384720295,
         ["430281807"] = 760.720454585032030,
         ["543966758"] = 820.481248322520287
      },

      ["10000"] = {
         ["670828913"] = 426.615290724301133,
         ["886885833"] = 553.987054063534742,
         ["715144259"] = 526.069033145121693,
         ["430281807"] = 440.124493104544513,
         ["543966758"] = 475.258634608114789
      }
   },

   ["model_8"] = {
      ["100"] = {
         ["670828913"] = 2935.691305839277447,
         ["886885833"] = 2932.533205710029051,
         ["715144259"] = 2936.835177109010147,
         ["430281807"] = 2933.440593883564361,
         ["543966758"] = 2929.779661751358162
      },

      ["1024"] = {
         ["670828913"] = 563.883584693520788,
         ["886885833"] = 378.294072284027266,
         ["715144259"] = 633.824942545395970,
         ["430281807"] = 576.890878573868463,
         ["543966758"] = 356.927908807500501 
      },

      ["10000"] = {
         ["670828913"] = 285.460492390177023,
         ["886885833"] = 348.496279190557289,
         ["715144259"] = 329.822709564486615,
         ["430281807"] = 292.391664710506916,
         ["543966758"] = 303.907183474437431
      }
   },

   ["model_9"] = {
      ["100"] = {
         ["670828913"] = 2947.228342295889433,
         ["886885833"] = 2957.943212612583011,
         ["715144259"] = 2928.134126839997862,
         ["430281807"] = 2935.745016790210229,
         ["543966758"] = 2952.513233804136689
      },

      ["1024"] = {
         ["670828913"] = 943.395765995267539,
         ["886885833"] = 957.958791248195439,
         ["715144259"] = 592.775686455428968,
         ["430281807"] = 835.298271207320909,
         ["543966758"] = 820.831710827126244
      },

      ["10000"] = {
         ["670828913"] = 122.206262838832771,
         ["886885833"] = 133.744975128874160,
         ["715144259"] = 118.458185627400283,
         ["430281807"] = 117.543148219584594,
         ["543966758"] = 130.284903444976180
      }
   },

   ["model_ninkovic"] = {
      ["100"] = {
         ["670828913"] = 2937.270545627479805,
         ["886885833"] = 2939.620581953350666,
         ["715144259"] = 2940.933239598960881,
         ["430281807"] = 2933.512897475653062,
         ["543966758"] = 2940.709278885540698
      },

      ["1024"] = {
         ["670828913"] = 888.503678718802689,
         ["886885833"] = 736.105441258723204,
         ["715144259"] = 849.104409256245958,
         ["430281807"] = 818.053938389397331,
         ["543966758"] = 863.885441187853075
      },

      ["10000"] = {
         ["670828913"] = 336.976034295859677,
         ["886885833"] = 451.341678694007328,
         ["715144259"] = 400.763683577053200,
         ["430281807"] = 346.564514785638266,
         ["543966758"] = 369.153317310450007
      }
   },


   ["model_triaxial"] = {
      ["100"] = {
         ["670828913"] = 2944.114097769592263,
         ["886885833"] = 2988.041868007017911,
         ["715144259"] = 2944.447789521596405,
         ["430281807"] = 2980.574335944069389,
         ["543966758"] = 2975.054300640168549
      },

      ["1024"] = {
         ["670828913"] = 1070.649088884108778,
         ["886885833"] = 1631.418068753951957,
         ["715144259"] = 957.175682293454429,
         ["430281807"] = 804.202399130802746,
         ["543966758"] = 980.598012865530450
      },

      ["10000"] = {
         ["670828913"] = 974.752775439882726,
         ["886885833"] = 1054.056790213983732,
         ["715144259"] = 940.619217975809647,
         ["430281807"] = 903.559235127309876,
         ["543966758"] = 779.164601359315043
      }
   },

   ["model_newhist1"] = {
      ["100"] = {
         ["670828913"] = 8332494.950073971413076,
         ["886885833"] = 6177679.766606546938419,
         ["715144259"] = 4731814.649677682667971,
         ["430281807"] = 3761930.196597724687308,
         ["543966758"] = 5653962.777469145134091
      },

      ["1024"] = {
         ["670828913"] = 3325353.230558808427304,
         ["886885833"] = 1089025.161034309305251,
         ["715144259"] = 2299149.824151862412691,
         ["430281807"] = 693459.971334934816696,
         ["543966758"] = 2509221.809957504738122
      },

      ["10000"] = {
         ["670828913"] = 2600.468131835904387,
         ["886885833"] = 2377.182166970440448,
         ["715144259"] = 2664.107721345673781,
         ["430281807"] = 2621.426890525223826,
         ["543966758"] = 2607.925069090109901
      }
   },

   ["model_newhist2"] = {
      ["100"] = {
         ["670828913"] = 4966887.924569396302104,
         ["886885833"] = 6944151.992310707457364,
         ["715144259"] = 6208271.237841151654720,
         ["430281807"] = 6336299.612545101903379,
         ["543966758"] = 5807816.585725879296660
      },

      ["1024"] = {
         ["670828913"] = 1992408.473517735488713,
         ["886885833"] = 1182742.498887087451294,
         ["715144259"] = 2635823.370834189467132,
         ["430281807"] = 1577120.227685633813962,
         ["543966758"] = 2035668.427606973098591
      },

      ["10000"] = {
         ["670828913"] = 830.154648887275471,
         ["886885833"] = 1251.552739880469744,
         ["715144259"] = 1294.715910959493613,
         ["430281807"] = 1062.547616702483538,
         ["543966758"] = 1046.946817049097035
      }
   },

   ["model_newhist3"] = {
      ["100"] = {
         ["670828913"] = 4695676.779752604663372,
         ["886885833"] = 4767228.354175671003759,
         ["715144259"] = 7315218.529508918523788,
         ["430281807"] = 4918005.005876600742340,
         ["543966758"] = 5988597.646148370578885
      },

      ["1024"] = {
         ["670828913"] = 1508035.348569449735805,
         ["886885833"] = 1850355.665624523768201,
         ["715144259"] = 3017760.004566484596580,
         ["430281807"] = 474424.234364059462678,
         ["543966758"] = 1085837.290311922086403
      },

      ["10000"] = {
         ["670828913"] = 2637.376575785131536,
         ["886885833"] = 2491.821929096237000,
         ["715144259"] = 2398.963300981807151,
         ["430281807"] = 2474.588894562217774,
         ["543966758"] = 2303.443184687581379
      }
   },

   ["model_LMC"] = {
      ["100"] = {
         ["670828913"] = 3156.823052563080637,
         ["886885833"] = 3170.567501216826258,
         ["715144259"] = 3122.488358157292168,
         ["430281807"] = 4918005.005876600742340,
         ["543966758"] = 5988597.646148370578885
      },

      ["1024"] = {
         ["670828913"] = 2561.577139944487044,
         ["886885833"] = 2951.514691699779632,
         ["715144259"] = 2844.622914280329951,
         ["430281807"] = 474424.234364059462678,
         ["543966758"] = 1085837.290311922086403
      },

      ["10000"] = {
         ["670828913"] = 3677.313271621078002,
         ["886885833"] = 3835.131744682992576,
         ["715144259"] = 3688.646043851602371,
         ["430281807"] = 2474.588894562217774,
         ["543966758"] = 2303.443184687581379
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


