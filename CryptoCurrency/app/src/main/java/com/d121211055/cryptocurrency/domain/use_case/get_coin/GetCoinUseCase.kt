package com.d121211055.cryptocurrency.domain.use_case.get_coin

import android.net.http.HttpException
import android.os.Build
import android.os.ext.SdkExtensions
import com.d121211055.cryptocurrency.common.Resource
import com.d121211055.cryptocurrency.data.remote.dto.toCoin
import com.d121211055.cryptocurrency.data.remote.dto.toCoinDetail
import com.d121211055.cryptocurrency.domain.model.Coin
import com.d121211055.cryptocurrency.domain.model.CoinDetail
import com.d121211055.cryptocurrency.domain.repository.CoinRepository
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import java.io.IOException
import javax.inject.Inject

class GetCoinUseCase @Inject constructor(
    private val repository: CoinRepository
) {
    operator fun invoke(coinId:String): Flow<Resource<CoinDetail>> = flow {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R && SdkExtensions.getExtensionVersion(
                Build.VERSION_CODES.S) >= 7) {
            try {
                emit(Resource.Loading<CoinDetail>())
                val coin = repository.getCoinById(coinId).toCoinDetail()
                emit(Resource.Success<CoinDetail>(coin))
            } catch (e:HttpException){
                emit(Resource.Error<CoinDetail>(e.localizedMessage ?: "An unexpected error occured"))
            } catch (e:IOException){
                emit(Resource.Error<CoinDetail>("Couldn't reach server. Check your internet connection"))
            }
        }
    }
}